import { Anthropic } from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import * as vscode from 'vscode';
import { ApiHandler, SingleCompletionHandler } from "../";
import { calculateApiCost } from "../../utils/cost";
import { ApiStream } from "../transform/stream";
import { convertToVsCodeLmMessages } from "../transform/vscode-lm-format";
import { SELECTOR_SEPARATOR, stringifyVsCodeLmModelSelector } from "../../shared/vsCodeSelectorUtils";
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api";

const ERROR_PREFIX = "Cline <Language Model API>";

interface TextBlock {
    type: string;
    text: string;
}

/**
 * Handles interaction with VS Code's Language Model API for chat-based operations.
 * This handler implements the ApiHandler interface to provide VS Code LM specific functionality.
 * 
 * @implements {ApiHandler}
 * 
 * @remarks
 * The handler manages a VS Code language model chat client and provides methods to:
 * - Create and manage chat client instances
 * - Stream messages using VS Code's Language Model API
 * - Retrieve model information
 * 
 * @example
 * ```typescript
 * const options = {
 *   vsCodeLmModelSelector: { vendor: "copilot", family: "gpt-4" }
 * };
 * const handler = new VsCodeLmHandler(options);
 * 
 * // Stream a conversation
 * const systemPrompt = "You are a helpful assistant";
 * const messages = [{ role: "user", content: "Hello!" }];
 * for await (const chunk of handler.createMessage(systemPrompt, messages)) {
 *   console.log(chunk);
 * }
 * ```
 */
export class VsCodeLmHandler implements ApiHandler, SingleCompletionHandler {

    private options: ApiHandlerOptions;
    private client: vscode.LanguageModelChat | null;
    private configurationWatcher: vscode.Disposable | null;
    private currentRequestCancellation: vscode.CancellationTokenSource | null;
    private systemPromptTokenCache: Map<string, number>;

    constructor(options: ApiHandlerOptions) {
        this.options = options;
        this.client = null;
        this.configurationWatcher = null;
        this.currentRequestCancellation = null;
        this.systemPromptTokenCache = new Map();

        try {
            this.configurationWatcher = vscode.workspace.onDidChangeConfiguration(event => {
                if (event.affectsConfiguration('lm')) {
                    this.releaseCurrentCancellation();
                    this.client = null;
                }
            });
        } catch (error) {
            this.dispose();
            throw new Error(
                `Cline <Language Model API>: Failed to initialize handler: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
        }
    }


    /**
     * Creates and streams a message using the VS Code Language Model API.
     *
     * @param systemPrompt - The system prompt to initialize the conversation context
     * @param messages - An array of message parameters following the Anthropic message format
     * 
     * @yields {ApiStream} An async generator that yields either text chunks or tool calls from the model response
     * 
     * @throws {Error} When vsCodeLmModelSelector option is not provided
     * @throws {Error} When the response stream encounters an error
     * 
     * @remarks
     * This method handles the initialization of the VS Code LM client if not already created,
     * converts the messages to VS Code LM format, and streams the response chunks.
     * Tool calls handling is currently a work in progress.
     */
    private releaseCurrentCancellation(): void {
        if (this.currentRequestCancellation) {
            this.currentRequestCancellation.cancel();
            this.currentRequestCancellation.dispose();
            this.currentRequestCancellation = null;
        }
    }

    async dispose(): Promise<void> {
        this.releaseCurrentCancellation();

        if (this.configurationWatcher) {
            this.configurationWatcher.dispose();
            this.configurationWatcher = null;
        }

        this.client = null;
        this.systemPromptTokenCache.clear();
    }

    private async countTokens(text: string | vscode.LanguageModelChatMessage): Promise<number> {
        // Check for required dependencies
        if (!this.client) {
            console.warn('Cline <Language Model API>: No client available for token counting');
            return 0;
        }

        if (!this.currentRequestCancellation) {
            console.warn('Cline <Language Model API>: No cancellation token available for token counting');
            return 0;
        }

        // Validate input
        if (!text) {
            console.debug('Cline <Language Model API>: Empty text provided for token counting');
            return 0;
        }

        try {
            // Handle different input types
            let tokenCount: number;

            if (typeof text === 'string') {
                tokenCount = await this.client.countTokens(text, this.currentRequestCancellation.token);
            } else if (text instanceof vscode.LanguageModelChatMessage) {
                // For chat messages, ensure we have content
                if (!text.content || (Array.isArray(text.content) && text.content.length === 0)) {
                    console.debug('Cline <Language Model API>: Empty chat message content');
                    return 0;
                }
                tokenCount = await this.client.countTokens(text, this.currentRequestCancellation.token);
            } else {
                console.warn('Cline <Language Model API>: Invalid input type for token counting');
                return 0;
            }

            // Validate the result
            if (typeof tokenCount !== 'number') {
                console.warn('Cline <Language Model API>: Non-numeric token count received:', tokenCount);
                return 0;
            }

            if (tokenCount < 0) {
                console.warn('Cline <Language Model API>: Negative token count received:', tokenCount);
                return 0;
            }

            return tokenCount;
        }
        catch (error) {
            // Handle specific error types
            if (error instanceof vscode.CancellationError) {
                console.debug('Cline <Language Model API>: Token counting cancelled by user');
                return 0;
            }

            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.warn('Cline <Language Model API>: Token counting failed:', errorMessage);

            // Log additional error details if available
            if (error instanceof Error && error.stack) {
                console.debug('Token counting error stack:', error.stack);
            }

            return 0; // Fallback to prevent stream interruption
        }
    }

    private async calculateInputTokens(systemPrompt: string, messages: any[]): Promise<number> {
        let totalTokens = 0;
        const systemPromptHash = createHash("sha1").update(systemPrompt).digest("base64");
        
        if (!this.systemPromptTokenCache.has(systemPromptHash)) {
            const tokenCount = await this.countTokens(systemPrompt);
            this.systemPromptTokenCache.set(systemPromptHash, tokenCount);
        }
        totalTokens += this.systemPromptTokenCache.get(systemPromptHash)!;

        for (const msg of messages) {
            if (msg.tokenCount !== undefined) {
                totalTokens += msg.tokenCount;
            } else {
                const messageContent = Array.isArray(msg.content)
                    ? msg.content.filter((block: TextBlock) => block.type === "text").map((block: TextBlock) => block.text).join("\n")
                    : msg.content;
                const tokenCount = await this.countTokens(messageContent);
                totalTokens += tokenCount;
            }
        }

        return totalTokens;
    }

    private async getClient(): Promise<vscode.LanguageModelChat> {
        if (!this.options.vsCodeLmModelSelector) {
            throw new Error(`${ERROR_PREFIX} The 'vsCodeLmModelSelector' option is required for the 'vscode-lm' provider.`);
        }

        if (!this.client) {
            try {
                this.client = await this.selectBestModel(this.options.vsCodeLmModelSelector);
            } catch (error) {
                const message = error instanceof Error ? error.message : 'Unknown error';
                console.error('Cline <Language Model API>: Client creation failed:', message);
                throw new Error(`${ERROR_PREFIX} Failed to create client: ${message}`);
            }
        }

        if (!this.client) {
            throw new Error(`${ERROR_PREFIX} Failed to initialize language model client`);
        }

        return this.client;
    }

    private async selectBestModel(selector: vscode.LanguageModelChatSelector): Promise<vscode.LanguageModelChat> {
        const models = await vscode.lm.selectChatModels(selector);
        if (models.length === 0) {
            throw new Error(`${ERROR_PREFIX} No models found matching the specified selector.`);
        }

        return models.reduce((best, current) =>
            current.maxInputTokens > best.maxInputTokens ? current : best,
            models[0]
        );
    }

    private cleanTerminalOutput(text: string): string {
        if (!text) {
            return '';
        }

        return text
            // Нормализуем переносы строк
            .replace(/\r\n/g, '\n')
            .replace(/\r/g, '\n')

            // Удаляем ANSI escape sequences
            .replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, '') // Полный набор ANSI sequences
            .replace(/\x9B[0-?]*[ -/]*[@-~]/g, '')  // CSI sequences

            // Удаляем последовательности установки заголовка терминала и прочие OSC sequences
            .replace(/\x1B\][0-9;]*(?:\x07|\x1B\\)/g, '')

            // Удаляем управляющие символы
            .replace(/[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F]/g, '')

            // Удаляем escape-последовательности VS Code
            .replace(/\x1B[PD].*?\x1B\\/g, '')      // DCS sequences
            .replace(/\x1B_.*?\x1B\\/g, '')         // APC sequences
            .replace(/\x1B\^.*?\x1B\\/g, '')        // PM sequences
            .replace(/\x1B\[[\d;]*[HfABCDEFGJKST]/g, '') // Cursor movement and clear screen

            // Удаляем пути Windows и служебную информацию
            .replace(/^(?:PS )?[A-Z]:\\[^\n]*$/mg, '')
            .replace(/^;?Cwd=.*$/mg, '')

            // Очищаем экранированные последовательности
            .replace(/\\x[0-9a-fA-F]{2}/g, '')
            .replace(/\\u[0-9a-fA-F]{4}/g, '')

            // Финальная очистка
            .replace(/\n{3,}/g, '\n\n')  // Убираем множественные пустые строки
            .trim();
    }

    private cleanMessageContent(content: any): any {
        if (!content) {
            return content;
        }

        if (typeof content === 'string') {
            return this.cleanTerminalOutput(content);
        }

        if (Array.isArray(content)) {
            return content.map(item => this.cleanMessageContent(item));
        }

        if (typeof content === 'object') {
            const cleaned: any = {};
            for (const [key, value] of Object.entries(content)) {
                cleaned[key] = this.cleanMessageContent(value);
            }
            return cleaned;
        }

        return content;
    }

    private async *processStreamChunks(response: vscode.LanguageModelChatResponse, contentBuilder: string[]): ApiStream {
        const stream = response.stream;

        for await (const chunk of stream) {
            if (this.currentRequestCancellation?.token.isCancellationRequested) {
                break;
            }

            if (chunk instanceof vscode.LanguageModelTextPart && chunk.value) {
                contentBuilder.push(chunk.value);
                yield { type: "text", text: chunk.value };
            }
        }

        this.releaseCurrentCancellation();
    }

    async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
        this.releaseCurrentCancellation();
        const client = await this.getClient();

        // Clean system prompt and messages
        const cleanedSystemPrompt = this.cleanTerminalOutput(systemPrompt);
        const cleanedMessages = messages.map(msg => ({
            ...msg,
            content: this.cleanMessageContent(msg.content)
        }));
        
        const vsCodeLmMessages = [
            vscode.LanguageModelChatMessage.Assistant(cleanedSystemPrompt),
            ...convertToVsCodeLmMessages(cleanedMessages)
        ];

        this.currentRequestCancellation = new vscode.CancellationTokenSource();
        const totalInputTokens = await this.calculateInputTokens(systemPrompt, messages);

        try {
            const contentBuilder: string[] = [];
            const response = await client.sendRequest(
                vsCodeLmMessages,
                {
                    justification: `Cline would like to use '${client.name}' from '${client.vendor}'. Click 'Allow' to proceed.`
                },
                this.currentRequestCancellation.token
            );

            const streamGenerator = this.processStreamChunks(response, contentBuilder);
            for await (const chunk of streamGenerator) {
                yield chunk;
            }

            if (!this.currentRequestCancellation?.token.isCancellationRequested) {
                const outputTokens = await this.countTokens(contentBuilder.join(""));
                yield {
                    type: "usage",
                    inputTokens: totalInputTokens,
                    outputTokens,
                    totalCost: calculateApiCost(
                        this.getModel().info,
                        totalInputTokens,
                        outputTokens
                    )
                };
            }
        } catch (error) {
            this.releaseCurrentCancellation();

            if (error instanceof vscode.CancellationError) {
                throw new Error(`${ERROR_PREFIX}: Request cancelled by user`);
            }

            if (error instanceof Error) {
                throw error;
            }

            throw new Error(`${ERROR_PREFIX}: Response stream error: ${String(error)}`);
        }
    }

    // Return model information based on the current client state
    getModel(): { id: string; info: ModelInfo; } {
        if (this.client) {
            // Validate client properties
            const requiredProps = {
                id: this.client.id,
                vendor: this.client.vendor,
                family: this.client.family,
                version: this.client.version,
                maxInputTokens: this.client.maxInputTokens
            };

            // Log any missing properties for debugging
            for (const [prop, value] of Object.entries(requiredProps)) {
                if (!value && value !== 0) {
                    console.warn(`Cline <Language Model API>: Client missing ${prop} property`);
                }
            }

            // Construct model ID using available information
            const modelParts = [
                this.client.vendor,
                this.client.family,
                this.client.version
            ].filter(Boolean);

            const modelId = this.client.id || modelParts.join(SELECTOR_SEPARATOR);

            // Build model info with conservative defaults for missing values
            const modelInfo: ModelInfo = {
                maxTokens: -1, // Unlimited tokens by default
                contextWindow: typeof this.client.maxInputTokens === 'number'
                    ? Math.max(0, this.client.maxInputTokens)
                    : openAiModelInfoSaneDefaults.contextWindow,
                supportsImages: false, // VSCode Language Model API currently doesn't support image inputs
                supportsPromptCache: true,
                inputPrice: 0,
                outputPrice: 0,
                description: `VSCode Language Model: ${modelId}`
            };

            return { id: modelId, info: modelInfo };
        }

        // Fallback when no client is available
        const fallbackId = this.options.vsCodeLmModelSelector
            ? stringifyVsCodeLmModelSelector(this.options.vsCodeLmModelSelector)
            : "vscode-lm";

        console.debug('Cline <Language Model API>: No client available, using fallback model info');

        return {
            id: fallbackId,
            info: {
                ...openAiModelInfoSaneDefaults,
                description: `VSCode Language Model (Fallback): ${fallbackId}`
            }
        };
    }

    async completePrompt(prompt: string): Promise<string> {
        try {
            const client = await this.getClient();
            const response = await client.sendRequest([vscode.LanguageModelChatMessage.User(prompt)], {}, new vscode.CancellationTokenSource().token);
            let result = "";
            for await (const chunk of response.stream) {
                if (chunk instanceof vscode.LanguageModelTextPart) {
                    result += chunk.value;
                }
            }
            return result;
        } catch (error) {
            if (error instanceof Error) {
                throw new Error(`VSCode LM completion error: ${error.message}`)
            }
            throw error
        }
    }
}
