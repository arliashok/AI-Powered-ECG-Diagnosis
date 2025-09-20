------------------------------------------------------------------
1.	Overview
------------------------------------------------------------------
This system implements an agentic loop:
1.	User supplies a natural-language task.
2.	A system prompt instructs the LLM to output a JSON list of tool invocations.
3.	The LLM (Azure OpenAI) returns a plan (array of tool steps).
4.	ActionExecutor parses and executes each step via the MCP HTTP server (mcp_server.py).
5.	Tool outputs are aggregated and returned to the user.
6.	Optional ECG deep-learning inference or GUI visualization is available as tools.
Core integration points:
•	Planning: LLM produces structured plan (no natural language answer expected).
•	Execution: Tools exposed over HTTP endpoints (pseudo-MCP style discovery + invoke routes).
•	Clinical Add-ons: ECG inference (inference.py) and signal processing / GUI (ECG_Processing.py).
------------------------------------------------------------------
2.	Module-by-Module Documentation
------------------------------------------------------------------
LLM_config.py Purpose: Centralized static configuration (currently hardcoded). Contents:
•	IDAM / OAuth config: IDAM_APP_CLIENT_ID, IDAM_APP_CLIENT_SECRET, IDAM_TOKEN_ENDPOINT, IDAM_LMAAS_APP_AUDIENCE
•	Azure OpenAI: OPENAI_ENDPOINT, OPENAI_DEPLOYMENT_MODEL, OPENAI_AZURE_API_VERSION
•	Bedrock placeholders (unused in current flow) Security Note: Secrets are hardcoded. Replace with environment variables and do not commit secrets. Example: import os IDAM_APP_CLIENT_ID = os.getenv("IDAM_APP_CLIENT_ID")
idam_token_generator.py Purpose: Obtain authenticated token for Azure OpenAI (via IDAM → exchange token). Key Methods:
•	get_unique_number(): Generates UUID used as a unique scope placeholder.
•	get_access_token(): Client credentials grant → base access token.
•	get_exchange_token(access_token): Exchanges for audience-specific token.
•	is_valid_token(token): Decodes JWT (no signature verification) and checks expiry.
•	get_idam_token(): Cache-aware retrieval or regeneration of fresh token.
•	write_token_to_cache(token) / get_token_from_cache(): Local JSON file cache. Risks:
•	No TLS verification (verify=False).
•	Cache file not permission-restricted.
•	Should handle exceptions with logging instead of print statements.
llm_client.py Purpose: Minimal async-friendly wrapper for Azure OpenAI chat completions. Data Classes:
•	LLMMessage(role, content)
•	LLMResponse(content) Functions:
•	get_azure_openai_direct_response(messages): Synchronous call to Azure OpenAI (uses IDAMTokenGenerator for bearer token).
•	generate_response(messages): Async wrapper using run_in_executor to avoid blocking event loop.
action_executor.py Purpose: Parse LLM output (expected JSON list) and sequentially call tools. Core Method:
•	parse_and_execute(llm_content, available_tools):
1.	json.loads → expects list of steps.
2.	For each step: { "tool": str, "parameters": dict }.
3.	Validates availability (names from discovery).
4.	Executes via mcp_client.invoke_tool.
5.	Aggregates success/error results. Returns: { "response": combined textual summary, "tool_results": [{tool, result|error}, ...] } Notes:
•	No concurrency (sequential). Could be extended with asyncio.gather.
•	No schema validation beyond presence of tool name.
mcp_client.py Purpose: Lightweight HTTP client for tool discovery and invocation. Functions:
•	discover_tools(): GET /.well-known/mcp.json
•	invoke_tool(tool_name, payload): POST /invoke/<tool> Assumptions:
•	Local server on port 5000. Improvements:
•	Add retries, timeouts, structured error translation.
mcp_server.py Purpose: Flask-based pseudo-MCP server exposing tools. Endpoints:
•	/.well-known/mcp.json: Returns static tool registry: get_patient_data, run_diagnostic_test, process_ecg, launch_ecg_gui
•	/invoke/get_patient_data: Returns mock demographics.
•	/invoke/run_diagnostic_test: Returns mock diagnostic status.
•	/invoke/process_ecg: Calls inference.predict_ecg(record) capturing stdout, parses ground truth & predicted labels from print lines.
•	/invoke/launch_ecg_gui: Spawns separate process running ECG_Processing.py (GUI). Utilities:
•	_is_display_available(): Prevent Tk launch in headless Linux. Design Notes:
•	Not a full MCP implementation (no JSON-RPC). Acts as a simplified HTTP tool layer.
•	Standardization layer could be added for consistent error envelopes.
agentic_ai_system.py Purpose: Orchestrates LLM planning + tool execution + memory. Data Classes:
•	AgentMemory: short_term, long_term, conversation_history, execution_history
•	AgentContext: goal, current_task, available_tools, resources, constraints, metadata Class: AgenticAISystem Key Methods:
•	__init__(): Initialize memory, context, executor, session_id, discover tools (_setup_tools()).
•	_setup_tools(): Calls mcp_client.discover_tools().
•	_build_system_prompt(): Instructs LLM to output JSON only (example plan included).
•	process_query(query):
1.	Update context + log user message.
2.	Build system prompt.
3.	Call generate_response.
4.	Execute resulting plan via ActionExecutor.
5.	Append assistant response to memory.
6.	Return consolidated response + tool results. Limitations:
•	No reflection loop.
•	No iterative refinement if LLM output malformed (first-failure returns raw content).
•	Assumes LLM always obeys JSON instruction.
main.py Purpose: CLI application loop. Class: AgenticAIClient Key Methods:
•	start_mcp_server(): (Currently commented out) Optionally spawn server subprocess.
•	initialize(): Create AgenticAISystem.
•	run_interactive_session(): REPL loop (quit/help/status/clear).
•	_print_* helpers for UI.
•	close(): Shutdown agent and terminate spawned server if any. Entrypoint:
•	asyncio.run(main()) launches interactive agent.
inference.py Purpose: Load PTB-XL metadata + trained model, run multilabel ECG classification. Workflow:
1.	Load CSV metadata: ptbxl_database.csv, scp_statements.csv.
2.	Build diagnostic labels (super/sub classes).
3.	Load Keras model from models/First_Paper.h5.
4.	Fit MultiLabelBinarizer from global label set. Functions:
•	aggregate_superclass_diagnostic(y_dic), aggregate_subclass_diagnostic(y_dic): Extract label taxonomies.
•	preprocess_ecg(record_name): Read WFDB sample, transpose, reshape (1, leads, 1000, 1).
•	predict_ecg(record_name): Preprocess, find ground truth by file name match, threshold predictions (>0.5), print results.
•	evaluate_model(): Load saved test arrays, reshape, evaluate model (expects specific metrics). Design Notes:
•	Prints output (parsed later by mcp_server).
•	No GPU placement logic; assumes environment supports TF.
ECG_Processing.py Purpose: Denoise, segment, label PQRST complexes, and visualize via Tkinter + Matplotlib. Key Components:
•	Data Load: Single record from PTB-XL (hardcoded path). Functions:
•	denoise_all_leads(ecg, fs, leads): Multi-stage pipeline:
1.	High-pass + wavelet baseline fusion
2.	Notch filters (50/100/150 Hz)
3.	Wavelet detail suppression + adaptive soft threshold
4.	Savitzky-Golay smoothing (if length adequate)
•	detect_pqrst(signal, fs): Uses wfdb.processing.XQRS for R detection; heuristics for P,Q,S,T windows.
•	segment_complexes(signal, r_locs, fs): Window extraction around R peaks with zero padding.
•	update_plot(...): Refresh all 2D/3D visualizations when lead changes. GUI Elements:
•	Tk root window, combobox for lead selection.
•	Three Matplotlib canvases: (Raw vs Denoised), (Denoised + PQRST markers), (3D PQRST complexes). Notes:
•	Blocking mainloop (intended for standalone use).
•	Not integrated into agent directly except via external process launch tool.
templates/index.html (present but not actively used by provided code)
•	Simple HTML form for query submission + result display (unused Flask UI in current server code).
------------------------------------------------------------------
3.	End-to-End Data / Control Flow
------------------------------------------------------------------
High-Level Steps:
1.	User enters query in CLI (main.py).
2.	AgenticAISystem.process_query builds system prompt listing discovered tools.
3.	Azure OpenAI returns JSON plan (list of steps).
4.	ActionExecutor parses each step and calls mcp_client.invoke_tool.
5.	Flask server dispatches the corresponding handler.
6.	Tool JSON results aggregated and returned.
7.	Response printed.
------------------------------------------------------------------
4.	Mermaid Architecture Diagram
------------------------------------------------------------------

graph TD
  U[User CLI Input] --> M[main.py / AgenticAIClient]
  M --> A[AgenticAISystem]
  A --> P[System Prompt Builder]
  A --> LLM[Azure OpenAI via llm_client.py]
  LLM --> A
  A -->|Plan JSON| AE[ActionExecutor]
  AE --> MC[mcp_client.py]
  MC --> S[mcp_server.py (Flask)]
  S --> T1[get_patient_data]
  S --> T2[run_diagnostic_test]
  S --> T3[process_ecg -> inference.predict_ecg]
  S --> T4[launch_ecg_gui -> ECG_Processing.py]
  T3 --> INF[inference.py (TF Model)]
  AE -->|Aggregated Results| A
  A --> OUT[Formatted Response]
  OUT --> U

------------------------------------------------------------------
5.	Mermaid Sequence: Query Processing & Tool Execution
------------------------------------------------------------------
sequenceDiagram
  participant User
  participant CLI as AgenticAIClient
  participant Agent as AgenticAISystem
  participant LLM as Azure OpenAI
  participant Exec as ActionExecutor
  participant MCP as mcp_client
  participant Server as mcp_server
  User->>CLI: Enter query
  CLI->>Agent: process_query(query)
  Agent->>Agent: _build_system_prompt()
  Agent->>LLM: messages (system+user)
  LLM-->>Agent: JSON plan (list of steps)
  Agent->>Exec: parse_and_execute(plan)
  loop For each step
    Exec->>MCP: invoke_tool(tool, params)
    MCP->>Server: POST /invoke/<tool>
    Server-->>MCP: JSON result
    MCP-->>Exec: result
  end
  Exec-->>Agent: combined response
  Agent-->>CLI: response + tool_results
  CLI-->>User: Display output

------------------------------------------------------------------
6.	Mermaid Flow: ECG Inference Tool (process_ecg)
------------------------------------------------------------------

flowchart TD
  A[POST /invoke/process_ecg] --> B{file_name provided?}
  B -- No --> E[Return 400 error]
  B -- Yes --> C[Validate inference module & predict_ecg]
  C --> D[Capture stdout]
  D --> F[Parse Ground Truth / Predicted Labels]
  F --> G[Return JSON: status, labels, raw_output]

------------------------------------------------------------------
7.	Mermaid Flow: ECG GUI Launch
------------------------------------------------------------------
flowchart TD
  A[POST /invoke/launch_ecg_gui] --> B{DISPLAY available?}
  B -- No --> X[Return 500 (no display)]
  B -- Yes --> C[Check ECG_Processing.py exists]
  C -- Missing --> Y[Return 500 error]
  C -- OK --> D[Spawn subprocess python ECG_Processing.py]
  D --> E[Return PID + status]

------------------------------------------------------------------
8.	Mermaid Flow: IDAM Token Retrieval
------------------------------------------------------------------
flowchart LR
  A[get_idam_token] --> B[Read cache]
  B --> C{Token valid?}
  C -- Yes --> Z[Return cached token]
  C -- No --> D[get_access_token()]
  D --> E[get_exchange_token()]
  E --> F[write_token_to_cache]
  F --> Z[Return new token]
------------------------------------------------------------------
9.	Mermaid Flow: PQRST Detection Pipeline (Simplified)
------------------------------------------------------------------
flowchart TD
  A[Load WFDB Record] --> B[Denoise Leads (multi-stage)]
  B --> C[Detect R Peaks (XQRS)]
  C --> D[Infer Q/S around R]
  D --> E[Infer P before Q]
  E --> F[Infer T after S]
  F --> G[Segment Complex Windows]
  G --> H[Visualize 2D/3D + Annotate]
------------------------------------------------------------------
10.	Key Design Considerations & Extensions
------------------------------------------------------------------
Reliability:
•	Missing retry/backoff for HTTP/LLM calls.
•	Single-pass plan execution (no validation/refinement loop).
Security:
•	Hardcoded credentials must be externalized.
•	Add TLS verification for IDAM calls (verify=True).
•	Sanitize any future command execution tools.
Scalability:
•	Tool calls currently sequential; can parallelize steps without dependencies.
•	Add schema validation (e.g., with pydantic) for tool parameters.
Resilience:
•	If LLM returns non-JSON, system returns raw text (no fallback re-prompt).
•	Could add auto-repair prompt to reformat invalid JSON.
Observability:
•	Add structured logging (JSON logs) and correlation IDs (session_id already exists).
•	Capture latency metrics per tool.
Memory / Context:
•	AgentMemory placeholders exist but not leveraged for reasoning loops.
•	Could implement summarization or retrieval of past actions for iterative tasks.
ECG ML:
•	Predictions threshold fixed at 0.5; consider calibration or top-k selection.
•	Add error handling for missing model file.
•	GUI decoupled; could expose derived analytics as additional API tool.
------------------------------------------------------------------
11.	Suggested Future Tools
------------------------------------------------------------------
•	summarize_results
•	retry_step
•	fetch_ecg_metadata
•	evaluate_model_subset
•	store_memory_item
------------------------------------------------------------------
12.	Risks / Gaps
------------------------------------------------------------------
•	No authentication on MCP endpoints (local only assumption).
•	GUI spawning may hang if environment lacks display (handled partially).
•	Model loading occurs at import time (cold start delay).
•	No rate limiting or concurrency guard on model inference.
------------------------------------------------------------------
13.	Quick Reference: Primary Runtime Call Graph
------------------------------------------------------------------
User → AgenticAIClient.run_interactive_session()
→ AgenticAISystem.process_query()
→ generate_response() (Azure OpenAI)
→ ActionExecutor.parse_and_execute()
→ mcp_client.invoke_tool() → mcp_server endpoint(s)
→ (Optional) inference.predict_ecg() or GUI spawn
→ Aggregate results → Return to user.
------------------------------------------------------------------
14.	Extension Strategy
------------------------------------------------------------------
To add a new tool:
1.	Implement Flask endpoint (or extend existing).
2.	Add tool descriptor in discovery JSON.
3.	LLM will start using it once discovered.
4.	Optionally enhance prompt with usage examples.
To add memory-based reasoning:
•	Extend _build_system_prompt() with last N conversation/tool outcomes.
•	Implement reflection pass: ask LLM if further steps required.
To add retry for invalid JSON:
•	Detect parse failure → send repair prompt: "Rewrite the previous plan strictly as JSON list..."
------------------------------------------------------------------
15.	Concise File Responsibility Matrix
------------------------------------------------------------------
•	main.py: CLI session + lifecycle.
•	agentic_ai_system.py: Orchestration, memory, prompt, execution.
•	llm_client.py: LLM transport (Azure OpenAI).
•	action_executor.py: Plan parsing & tool dispatch.
•	mcp_client.py: HTTP client for tools.
•	mcp_server.py: Tool registry + endpoints (mock + ECG + GUI).
•	inference.py: Deep learning ECG classification.
•	ECG_Processing.py: Signal processing + visualization.
•	idam_token_generator.py: Token acquisition & caching.
•	LLM_config.py: Static configuration (should be env-based).
•	templates/index.html: Unused sample HTML form (potential web UI).
------------------------------------------------------------------
16.	Validation & Testing Suggestions
------------------------------------------------------------------
Unit:
•	Mock Azure OpenAI + test malformed JSON recovery.
•	Test ActionExecutor with valid/invalid tools.
Integration:
•	Spin up Flask server; run end-to-end query producing multi-step tool plan.
ML:
•	Add shape tests for preprocess_ecg.
•	Add test ensuring predict_ecg outputs deterministic label set for fixture input (mock model).
------------------------------------------------------------------
17.	Known Improvement Opportunities
------------------------------------------------------------------
•	Introduce proper MCP JSON-RPC protocol compliance.
•	Add dependency graph resolution (parallel tool dispatch where possible).
•	Add structured result objects instead of concatenated string summary.
•	Implement configurable planning vs execution separation (plan preview step).
------------------------------------------------------------------

------------------------------------------------------------------
1.	High-Level Architectural View
------------------------------------------------------------------
Layers:
1.	Interface: CLI (main.py) optionally future web UI.
2.	Orchestration: AgenticAISystem builds prompt, invokes LLM, dispatches plan.
3.	Planning Model: Azure OpenAI (chat completion) returns JSON tool plan.
4.	Execution: ActionExecutor validates & executes steps via HTTP tool layer.
5.	Tool Transport: Simplified MCP-style HTTP (mcp_client.py <-> mcp_server.py).
6.	Domain Tools: Patient mock data, diagnostic mock, ECG inference, ECG GUI.
7.	ML Subsystem: inference.py (TF model + label taxonomy).
8.	Signal Visualization: ECG_Processing.py (denoise, detect, segment, visualize).
Core Data Contracts:
•	LLM Planning Output (expected): [ {"tool": "<tool_name>", "parameters": { ... }}, ... ]
•	Tool Discovery JSON: { "tools": [ { "name": str, "description": str, "parameters": {<k>: "string"...} }, ... ] }
•	Tool Invocation Request: POST /invoke/<tool>  { paramKey: value }
•	Tool Invocation Response: JSON (tool-specific result) or HTTP 4xx/5xx with {error: "..."}.
------------------------------------------------------------------
2.	Detailed Module Documentation
------------------------------------------------------------------
2.1 main.py Purpose: User-facing CLI loop controlling lifecycle. Key Classes/Functions:
•	AgenticAIClient:
•	initialize(): Creates AgenticAISystem. (Optionally could start server; line commented.)
•	run_interactive_session(): Blocking REPL. Commands: help/status/clear/quit.
•	start_mcp_server(): Spawns Flask server (currently unused—assumes external run).
•	close(): Graceful teardown, terminates server subprocess if started. Flow: input() → system.process_query() → print response/tool summaries. Edge Cases:
•	KeyboardInterrupt caught for graceful exit.
•	If server not running, tool discovery yields empty tool list; LLM prompt fallback includes default tool names. Improvements:
•	Add async streaming for partial responses.
•	Add argument parsing (e.g., --auto-start-server).
2.2 agentic_ai_system.py Purpose: Core orchestrator (memory + context + LLM call + tool execution). Data Classes:
•	AgentMemory: Provides structural placeholders; not yet leveraged for retrieval or summarization.
•	AgentContext: Tracks current goal, available tools. Class: AgenticAISystem Methods:
•	__init__(): Initializes memory, context, session_id, ActionExecutor, discovers tools.
•	_setup_tools(): HTTP fetch via discovery; stores names only.
•	_build_system_prompt(): Strict instruction to return JSON list (includes example). Risk: Example missing commas (third line missing comma before next object) may cause hallucination; should be validated.
•	process_query(query):
1.	Logs & records user message.
2.	Builds message array (system + user).
3.	Calls generate_response.
4.	Executes tool plan (if JSON parse succeeds).
5.	Records assistant response.
6.	Returns unified response + tool_results. Error Handling:
•	If JSON parse fails, returns original LLM text as plain response (no repair loop). Potential Enhancements:
•	Add JSON schema validation & auto-repair prompt.
•	Add multi-iteration planning (evaluate results → new plan).
•	Integrate memory summarization and retrieval.
2.3 llm_client.py Purpose: Simple Azure OpenAI wrapper. Key Functions:
•	get_azure_openai_direct_response(messages: List[dict]):
•	Builds IDAM token each call (no token reuse across calls in this function—IDAMTokenGenerator caches though).
•	Calls AzureOpenAI.chat.completions.create.
•	Returns first candidate content.
•	generate_response(messages: List[LLMMessage]):
•	Converts dataclasses → dict.
•	Executes sync call in thread executor (non-blocking to event loop). Constraints:
•	No temperature parameterization externally.
•	No tool function calling / function schema support. Improvements:
•	Add retries on rate limits.
•	Abstract provider strategy.
2.4 idam_token_generator.py Purpose: Token acquisition with caching. Important Methods:
•	get_idam_token(): Orchestrates cache retrieval and regeneration. Security & Reliability Issues:
•	verify=False in requests (disables TLS validation).
•	Plaintext secrets and tokens on disk.
•	No concurrency lock on cache file.
•	Print statements (should use logging). Mitigations:
•	Use environment variables + proper TLS.
•	Add file permission hardening.
•	Apply in-memory + timed refresh.
2.5 action_executor.py Purpose: Deterministic execution of plan steps. Method: parse_and_execute(llm_content, available_tools) Steps:
1.	JSON parse (strict).
2.	Validate list shape.
3.	For each step:
•	Validate tool membership.
•	Invoke via mcp_client.invoke_tool.
4.	Aggregate human-readable summary (string). Edge Cases:
•	Missing "parameters" defaults to {}.
•	Non-existent tool logged as warning. Enhancements:
•	Add concurrency for independent steps.
•	Support conditional branching or output variable passing.
•	Replace string concatenation with structured result object.
2.6 mcp_client.py Purpose: Thin HTTP wrapper (no MCP protocol semantics beyond discovery). Functions:
•	discover_tools(): Returns list of tool definitions.
•	invoke_tool(tool_name, payload): POST JSON payload. Improvements:
•	Add timeout, retry (e.g., requests.Session + backoff).
•	Standardize exceptions (wrap in custom error class).
•	Add response schema validation.
2.7 mcp_server.py Purpose: Tool hosting pseudo-MCP service. Endpoints:
•	/.well-known/mcp.json: Static tool registry.
•	/invoke/get_patient_data: Returns mock patient record (echo of patient_id).
•	/invoke/run_diagnostic_test: Returns static "Normal" result.
•	/invoke/process_ecg: Flow: a. Validate file_name. b. Ensure inference imported and has predict_ecg. c. Capture stdout of predict_ecg. d. Parse lines containing "Ground Truth Labels:" and "Predicted Labels". e. Return parsed + raw output. Failure Modes:
•	400 if missing file_name.
•	500 if module import or prediction fails. Improvement:
•	Provide structured array of predicted labels instead of raw string parse.
•	/invoke/launch_ecg_gui: Flow: a. Validate display availability (Linux DISPLAY env). b. Check ECG_Processing.py exists. c. Optional record_path passed via env var. d. Spawn subprocess; returns PID. Risks:
•	Long-lived process accumulation; add cleanup endpoint.
•	Captured stdout/stderr unused; could stream logs. General Improvements:
•	Unify error schema: { "error": { "code": "...", "message": "...", "details": {...} } }.
•	Add health endpoint.
•	Implement JSON-RPC 2.0 if full MCP compliance desired.
2.8 inference.py Purpose: Load pretrained multi-label ECG model & enable single-record inference. Key Global Steps (executed on import):
•	Load PTB-XL metadata.
•	Build diagnostic label mapping (superclass/subclass).
•	Load Keras model.
•	Fit MultiLabelBinarizer. Functions:
•	preprocess_ecg(record_name): Reads WFDB sample, shapes to (1, leads, 1000, 1).
•	predict_ecg(record_name): a. Preprocess. b. Lookup ground truth in metadata. c. Model inference. d. Threshold > 0.5, inverse transform labels. e. Print ground truth & predicted.
•	evaluate_model(): Loads pre-saved arrays, evaluates model metrics. Performance:
•	Model load at module import: increases cold start latency.
•	Consider lazy loading or a warmup function. Robustness:
•	No exception handling around file I/O or model missing.
•	Hardcoded paths; parameterize via env.
2.9 ECG_Processing.py Purpose: Interactive visualization of ECG denoising, PQRST detection, segmentation. Pipeline Functions:
•	denoise_all_leads: Multi-stage wavelet + filter pipeline (baseline, notch, wavelet shrinkage, smoothing).
•	detect_pqrst: Uses XQRS for R, then window heuristics for others (may mislabel under noise).
•	segment_complexes: Extracts windowed complexes around each R with zero-padding. GUI:
•	Tkinter + Matplotlib embedded canvases.
•	Combobox switches selected lead.
•	3D plot shows multiple complexes stacked. Improvements:
•	Extract logic into pure functions for testability.
•	Use threads or async for expensive operations if scaling.
•	Accept external record path via environment variable (already partially scaffolded).
2.10 templates/index.html Purpose: Simple form-based UI (unused currently). Could be integrated by running a Flask client layer separate from tool server.
------------------------------------------------------------------
3.	Detailed Endpoint Contracts
------------------------------------------------------------------
GET /.well-known/mcp.json Response 200: { "tools": [ { "name": str, "description": str, "parameters": {param: "string"...} } ] }
POST /invoke/get_patient_data Request: { "patient_id": str } Response 200: { patient_id, name, age, history }
POST /invoke/run_diagnostic_test Request: { "patient_id": str } Response 200: { patient_id, status, result }
POST /invoke/process_ecg Request: { "file_name": str } Success 200: { "status": "ECG prediction executed", "file_name": str, "ground_truth": <list|string|null>, "predicted_labels": <tuple|string|null>, "raw_output": str } Errors: 400 missing parameter 500 inference import or runtime error
POST /invoke/launch_ecg_gui Request: { "record_path": optional str } Success 200: { status, pid, note, record_path_override } Errors: 500 (no display, missing script, spawn failure)
------------------------------------------------------------------
4.	Failure & Edge Case Matrix (Selected)
------------------------------------------------------------------
LLM returns non-JSON → raw text passed back, no execution. Tool name typo → recorded as "Tool not available". ECG file not found → inference.predict_ecg likely raises; 500 returned. Model file missing → load_model failure at import; subsequent process_ecg returns 500. Headless environment GUI launch → explicit 500 error.
------------------------------------------------------------------
5.	Security & Compliance Enhancements
------------------------------------------------------------------
•	Remove hardcoded secrets; adopt environment variables + dotenv file skeleton.
•	Enable TLS verification (verify=True).
•	Add authentication/authorization to tool endpoints (API key or JWT).
•	Sanitize any future command execution tools.
•	Add rate limiting (Flask-Limiter).
•	Restrict model path and file_name to whitelisted directory (avoid path traversal).
------------------------------------------------------------------
6.	Performance Opportunities
------------------------------------------------------------------
•	Lazy-load model on first ECG inference.
•	Parallelize independent tool steps (async gather).
•	Cache tool discovery result with TTL.
•	Reuse HTTP session (requests.Session) for lower connection overhead.
------------------------------------------------------------------
7.	Observability
------------------------------------------------------------------
Add:
•	Correlated request ID (session_id) in headers to server.
•	Structured JSON logging.
•	Timing metrics (per tool latency).
•	Basic health: GET /health (returns model loaded status, tool count).
------------------------------------------------------------------
8.	Extension Patterns
------------------------------------------------------------------
Adding a new tool:
1.	Implement Flask endpoint: /invoke/<tool_name>.
2.	Add descriptor object to discovery JSON.
3.	Ensure deterministic JSON output.
4.	(Optional) Add example usage to system prompt.
Adding plan validation:
•	Introduce JSON schema (e.g., with jsonschema) in ActionExecutor.
•	On failure, send corrective prompt: "Your previous response was invalid JSON. Return ONLY valid JSON list..."
Memory/Reflection:
•	Append execution results to execution_history.
•	Provide truncated summaries (e.g., last 5 results) inside _build_system_prompt().
------------------------------------------------------------------
9.	Example Improved System Prompt (Optional Upgrade)
------------------------------------------------------------------
You are a planning assistant. Return ONLY a JSON array of steps. Each step: {"tool": "<name>", "parameters": { ... }}. Available tools: <dynamic>. Do not include commentary. If information must be gathered before further action, include those tool steps explicitly.
------------------------------------------------------------------
10.	Expanded Mermaid (Error Path for process_ecg)
------------------------------------------------------------------
flowchart TD
  A[POST /invoke/process_ecg] --> B{file_name empty?}
  B -- Yes --> E[400 JSON {error}]
  B -- No --> C{inference module loaded?}
  C -- No --> F[500 JSON {error, details}]
  C -- Yes --> G{predict_ecg callable?}
  G -- No --> H[500 {error}]
  G -- Yes --> I[Run predict_ecg capture stdout]
  I --> J{Exception?}
  J -- Yes --> K[500 {error}]
  J -- No --> L[Parse labels]
  L --> M[200 JSON payload]
------------------------------------------------------------------
11.	Code Quality Improvement Checklist
------------------------------------------------------------------
•	Replace print with logger across all modules.
•	Add type hints to all function signatures (some missing).
•	Extract magic constants (e.g., thresholds, window sizes) into config.
•	Unit-test parsing logic in process_ecg.
•	Harden JSON parsing with schema.
------------------------------------------------------------------
12.	Suggested Test Cases
------------------------------------------------------------------
ActionExecutor:
•	Valid single tool plan.
•	Invalid JSON (string) → returns raw text.
•	Tool not available → error aggregated.
MCP Server:
•	process_ecg missing file_name → 400.
•	process_ecg with fake file → simulate failure.
•	launch_ecg_gui headless → 500.
Inference:
•	preprocess_ecg shape correctness.
•	predict_ecg with mocked model (dependency injection / monkeypatch).
------------------------------------------------------------------
13.	Migration Toward Full MCP (If Desired)
------------------------------------------------------------------
Add:
•	JSON-RPC envelope: { "jsonrpc": "2.0", "id": <id>, "method": "tools/list", "params": {} }
•	Tool invocation: "method": "tools/call"
•	Standardized error object: { "code": int, "message": str, "data": any }
•	Session handshake & capability negotiation.
------------------------------------------------------------------
14.	Risks Summary
------------------------------------------------------------------
•	Secret leakage (hardcoded).
•	No auth on tool endpoints.
•	Malformed LLM output halts execution (no self-repair).
•	Model load failure not surfaced clearly to user query path.
•	Potential resource leak of GUI subprocesses.
------------------------------------------------------------------
15.	Minimal Refactor Priorities (If Acting Immediately)
------------------------------------------------------------------
1.	Externalize all secrets to env.
2.	Add JSON schema validation for plan.
3.	Add safe fallback prompt on parse failure.
4.	Introduce retries + timeouts for HTTP & LLM calls.
5.	Add lazy model load & failure reporting endpoint.
------------------------------------------------------------------
16.	Glossary
------------------------------------------------------------------
Plan: JSON list returned by LLM specifying tool sequence. Tool: HTTP endpoint representing an atomic operation. Memory: Persistent (not yet implemented fully) contextual state for reasoning. Inference: ML classification stage for ECG signals.
------------------------------------------------------------------
17.	Quick Reference (Function to Responsibility)
------------------------------------------------------------------
•	AgenticAISystem.process_query: End-to-end request handling.
•	ActionExecutor.parse_and_execute: Deterministic plan execution.
•	get_azure_openai_direct_response: Raw LLM call.
•	predict_ecg: Single-record multi-label classification.
•	denoise_all_leads: Multi-stage signal conditioning.
•	detect_pqrst: Peak identification heuristic.
•	launch_ecg_gui endpoint: Spawns visualization process.
------------------------------------------------------------------
