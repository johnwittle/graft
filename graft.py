#!/usr/bin/env python3
# TODO: Add actual enforcement of tool permissions in ToolExecutor.execute()
#       Currently the /shell flag only affects what tools are advertised to the
#       model, but a model that "believes" it has access (e.g. from reading source)
#       can still invoke tools. This is fine for trusted models (Claude) but
#       matters for future multi-model support.

"""
graft - A conversation harness for the Anthropic API

Manages persistent conversations with Claude, supporting save/load,
prompt caching, and seamless context continuity.

Usage: 
  graft                          # Start new conversation or interactive menu
  graft <name>                   # Load existing conversation by name
  graft --import <file.json> [name] [--no-thinking] [--no-tool-use]
                                 # Import from socketteer or API format
  graft --list                   # List saved conversations

Commands during conversation:
  /save [name]    - Save conversation (prompts for name if new)
  /load <name>    - Load a different conversation
  /list           - Show saved conversations
  /new            - Start fresh conversation
  /rename <name>  - Rename current conversation
  /delete <n>  - Delete a saved conversation
  /read [--tools] [--thinking] - View transcript in pager
  /export [--tools] [--thinking] [file] - Export transcript
  /cache on|off|5m|1h  - Control prompt caching
  /model [name]   - Show or switch model
  /max_tokens [n] - Show or set max output tokens
  /thinking [n|mode] [n]  - Extended thinking. See '/thinking' with no args for modes.
  /web on|off     - Toggle web search
  /tools [path]   - Enable file tools for a directory
  /tokens         - Show token estimates
  /system [text]  - Set/show system prompt
  /stats          - Show session statistics
  /help           - Show commands
  /quit           - Exit
"""

import json
import sys
import os
import readline
import atexit
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

# === Configuration ===

GRAFT_DIR = Path.home() / ".graft"
CONVERSATIONS_DIR = GRAFT_DIR / "conversations"
CONFIG_PATH = GRAFT_DIR / "config.toml"
HISTORY_PATH = GRAFT_DIR / "history"

DEFAULT_CONFIG = {
    "default_model": "claude-opus-4-5-20251101",
    "cache_ttl": "5m",
    "editing_mode": "emacs",
    "max_tokens": 8192,
    "thinking_budget": 0,  # 0 = disabled, >1024 = enabled
    "thinking_mode": "auto",  # 'auto' | 'adaptive' | 'enabled' | 'off' - see /thinking
    "web_search": False,
    "default_system_prompt": "",
}

# === Setup Functions ===

def ensure_graft_dirs():
    """Create ~/.graft directory structure if needed."""
    GRAFT_DIR.mkdir(exist_ok=True)
    CONVERSATIONS_DIR.mkdir(exist_ok=True)

def load_config():
    """Load config from TOML file, with defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if CONFIG_PATH.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Last-resort fallback: only handles simple single-line key=value.
                # Does not understand triple-quoted strings, booleans, arrays, etc.
                # Install `tomli` (pip install tomli) to get full TOML support on Python < 3.11.
                for line in CONFIG_PATH.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        if key in config:
                            config[key] = value
                return config

        with open(CONFIG_PATH, 'rb') as f:
            file_config = tomllib.load(f)
            config.update(file_config)

    return config

def save_default_config():
    """Write default config if none exists."""
    if not CONFIG_PATH.exists():
        config_text = '''# graft configuration

# Default model for new conversations
default_model = "claude-opus-4-5-20251101"

# Prompt cache TTL: "5m", "1h", or "off"
cache_ttl = "5m"

# Line editing mode: "vi" or "emacs"
editing_mode = "emacs"

# Maximum response tokens
max_tokens = 8192

# Extended thinking budget (0 = disabled, minimum 1024 when enabled)
thinking_budget = 0

# Extended thinking mode: "auto" (recommended - picks adaptive vs a literal
# token budget per the active model), "adaptive", "enabled" (force a literal
# budget where the model still allows it), or "off". See /thinking.
thinking_mode = "auto"

# Enable web search by default: true or false
# Costs $10 per 1,000 searches. Must be enabled in Anthropic Console.
web_search = false

# System prompt applied to new conversations. Use triple-quoted strings for
# multi-line prompts. Loaded conversations keep whatever prompt they were
# saved with; this only affects /new. Override in a conversation with /system.
default_system_prompt = ""
'''
        CONFIG_PATH.write_text(config_text)

def setup_readline(config):
    """Configure readline for line editing and history."""
    # Set editing mode
    if config.get('editing_mode', 'emacs').lower() == 'vi':
        readline.parse_and_bind('set editing-mode vi')
    else:
        readline.parse_and_bind('set editing-mode emacs')
    
    # Load history
    if HISTORY_PATH.exists():
        try:
            readline.read_history_file(HISTORY_PATH)
        except Exception:
            pass
    
    # Set history length
    readline.set_history_length(1000)
    
    # Save history on exit
    atexit.register(lambda: readline.write_history_file(HISTORY_PATH))

def load_dotenv():
    """Load .env file and return dict of values (without polluting os.environ)."""
    env_vars = {}
    for env_path in [Path('.env'), Path.home() / '.env', GRAFT_DIR / '.env']:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"\'')
            break
    return env_vars

# === Conversation Management ===

def convert_socketteer_via_import(path, include_thinking=True, include_tool_use=True):
    """
    Convert socketteer export by calling bin/graft-import subprocess.
    Returns (messages, name, model) tuple.
    """
    import subprocess

    # Find graft-import relative to this file
    bin_dir = Path(__file__).resolve().parent / 'bin' / 'graft-import'
    if not bin_dir.exists():
        raise FileNotFoundError(f"graft-import not found at {bin_dir}")

    cmd = [sys.executable, str(bin_dir), str(path)]
    if not include_thinking:
        cmd.append('--no-thinking')
    if not include_tool_use:
        cmd.append('--no-tool-use')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"graft-import failed: {result.stderr.strip()}")

    # Print graft-import's stderr (progress/warnings) to our stderr
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)

    data = json.loads(result.stdout)
    return data['messages'], data.get('name', 'imported'), data.get('model')


class Conversation:
    """Manages a single conversation's state and metadata."""
    
    def __init__(self, name=None):
        self.name = name
        self.messages = []
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self.model = None  # Set from config if not loaded
        self.system_prompt = ""
        self.unsaved_changes = False
        # Tool settings (persisted with conversation)
        self.web_search = False
        self.tools_path = None  # Path string if tools enabled
        self.shell_enabled = False
        self.thinking_budget = 0  # 0 = disabled
        self.thinking_mode = "auto"  # 'auto' | 'adaptive' | 'enabled' | 'off'
    
    @classmethod
    def load(cls, name):
        """Load conversation from ~/.graft/conversations/<name>.json"""
        path = CONVERSATIONS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No conversation named '{name}'")
        
        data = json.loads(path.read_text(encoding='utf-8'))
        
        conv = cls(name=data.get('name', name))
        conv.messages = data.get('messages', [])
        conv.created = data.get('created', datetime.now().isoformat())
        conv.modified = data.get('modified', conv.created)
        conv.model = data.get('model')
        conv.system_prompt = data.get('system_prompt', "")
        conv.unsaved_changes = False
        # Tool settings
        conv.web_search = data.get("web_search", False)
        conv.tools_path = data.get("tools_path", None)
        conv.shell_enabled = data.get("shell_enabled", False)
        conv.thinking_budget = data.get("thinking_budget", 0)
        conv.thinking_mode = data.get("thinking_mode", "auto")

        return conv
    
    @classmethod
    def from_import(cls, path, name=None, include_thinking=True, include_tool_use=True):
        """
        Import conversation from various formats:
        - socketteer export (has 'chat_messages')
        - json_to_api.py output (has 'messages' + 'metadata')
        - raw API messages array
        
        By default, includes thinking blocks and tool use for full continuity.
        Use --no-thinking or --no-tool-use to exclude.
        """
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        
        # Detect format and convert
        if isinstance(data, dict) and 'chat_messages' in data:
            # Socketteer format - delegate to bin/graft-import
            messages, default_name, source_model = convert_socketteer_via_import(
                path, include_thinking, include_tool_use
            )
        elif isinstance(data, dict) and 'messages' in data:
            # Already API format (from json_to_api.py or graft save)
            messages = data['messages']
            metadata = data.get('metadata', {})
            default_name = metadata.get('source_name', data.get('name', 'imported'))
            source_model = metadata.get('source_model', data.get('model'))
        elif isinstance(data, list):
            # Raw messages array
            messages = data
            default_name = 'imported'
            source_model = None
        else:
            raise ValueError("Unrecognized conversation format")
        
        conv = cls(name=name or default_name)
        conv.messages = messages
        conv.model = source_model
        conv.unsaved_changes = True
        
        return conv
    
    def save(self, new_name=None):
        """Save conversation to ~/.graft/conversations/<name>.json"""
        if new_name:
            self.name = new_name
        
        if not self.name:
            raise ValueError("Conversation needs a name to save")
        
        self.modified = datetime.now().isoformat()
        
        data = {
            'name': self.name,
            'created': self.created,
            'modified': self.modified,
            'model': self.model,
            'system_prompt': self.system_prompt,
            'messages': self.messages,
            'web_search': self.web_search,
            'tools_path': self.tools_path,
            'shell_enabled': self.shell_enabled,
            'thinking_budget': self.thinking_budget,
            'thinking_mode': self.thinking_mode,
        }
        
        # Sanitize name to prevent path traversal
        safe_name = self.name.replace('/', '_').replace('\\', '_').replace('\0', '')
        if not safe_name or safe_name.strip('.') == '':
            raise ValueError("Invalid conversation name")
        path = CONVERSATIONS_DIR / f"{safe_name}.json"
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self.unsaved_changes = False
        
        return path
    
    def token_estimate(self):
        """Rough token count estimate."""
        total = 0
        for m in self.messages:
            content = m.get('content', '')
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        total += len(block['text'])
        return total // 2  # ~2 chars per token for conversation text

def list_conversations():
    """Return list of saved conversations with metadata."""
    convos = []
    for path in sorted(CONVERSATIONS_DIR.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            messages = data.get('messages', [])
            convos.append({
                'name': path.stem,
                'modified': data.get('modified', 'unknown'),
                'messages': len(messages),
                'model': data.get('model', 'unknown'),
            })
        except Exception as e:
            convos.append({
                'name': path.stem,
                'error': str(e),
            })
    return convos

def format_conversation_list(convos):
    """Format conversation list for display."""
    if not convos:
        return "No saved conversations."
    
    lines = []
    for c in convos:
        if 'error' in c:
            lines.append(f"  {c['name']} (error: {c['error']})")
        else:
            # Parse and format the date
            try:
                dt = datetime.fromisoformat(c['modified'])
                date_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                date_str = c['modified'][:16] if len(c['modified']) > 16 else c['modified']
            
            lines.append(f"  {c['name']:<30} {c['messages']:>4} msgs  {date_str}")
    
    return '\n'.join(lines)

# === Transcript Formatting ===

def format_message(msg, include_tools=False, include_thinking=False):
    """Format a single message for display."""
    role = msg.get('role', 'unknown')
    content = msg.get('content', '')
    
    # Handle block-format content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'image':
                    text_parts.append("[image]")
                elif 'text' in block:
                    text_parts.append(block['text'])
                elif include_thinking and block.get('type') == 'thinking':
                    thinking_content = block.get('thinking', '')
                    text_parts.append(f"[Thinking]\n{thinking_content}\n[/Thinking]")
                elif include_tools and block.get('type') == 'tool_use':
                    tool_name = block.get('name', 'unknown')
                    tool_input = block.get('input', {})
                    input_str = json.dumps(tool_input)
                    if len(input_str) > 100:
                        input_str = input_str[:100] + '...'
                    text_parts.append(f"[Tool: {tool_name}({input_str})]")
                elif include_tools and block.get('type') == 'tool_result':
                    tool_content = block.get('content', '')
                    # Truncate long results
                    if len(tool_content) > 200:
                        tool_content = tool_content[:200] + '...'
                    text_parts.append(f"[Result: {tool_content}]")
        content = '\n\n'.join(text_parts)
    
    # Format role header
    if role == 'user':
        header = "You:"
    elif role == 'assistant':
        header = "Claude:"
    else:
        header = f"{role.title()}:"
    
    return f"{header}\n{content}"

def format_transcript(messages, last_n=None, include_tools=False, include_thinking=False):
    """Format messages as human-readable transcript."""
    if last_n:
        messages = messages[-last_n:]
    
    parts = []
    for msg in messages:
        parts.append(format_message(msg, include_tools=include_tools, include_thinking=include_thinking))
    
    separator = "\n\n" + ("-" * 40) + "\n\n"
    return separator.join(parts)

def show_in_pager(text):
    """Display text in a pager (less) if available, otherwise print."""
    import subprocess
    import shutil
    
    # Try to find a pager
    pager = shutil.which('less') or shutil.which('more')
    
    if pager:
        try:
            proc = subprocess.Popen([pager], stdin=subprocess.PIPE)
            proc.communicate(input=text.encode('utf-8'))
            return
        except Exception:
            pass
    
    # Fallback: just print
    print(text)

def show_recent_messages(messages, n=4):
    """Print the last n messages as context."""
    if not messages:
        return
    
    recent = messages[-n:]
    print("\n--- Recent context ---")
    for msg in recent:
        print()
        print(format_message(msg))
    print("\n" + "-" * 40)

# === Prompt Caching ===

def strip_unsigned_thinking_blocks(messages):
    """
    Remove thinking blocks that lack valid signatures.
    
    Thinking block signatures are cryptographic verification that the block
    was generated by Claude. Blocks imported from external sources (like
    socketteer exports) don't have signatures and cannot be sent to the API.
    
    We keep them in storage for /read --thinking, but must filter here.
    Blocks with valid signatures (from graft's own extended thinking) are kept.
    """
    result = []
    for msg in messages:
        content = msg.get('content')
        
        if isinstance(content, list):
            # Filter out unsigned thinking blocks
            filtered = [
                b for b in content 
                if not (isinstance(b, dict) and 
                       b.get('type') == 'thinking' and 
                       not b.get('signature'))
            ]
            if filtered:
                # Simplify to string if only text blocks remain
                if len(filtered) == 1 and filtered[0].get('type') == 'text':
                    result.append({'role': msg['role'], 'content': filtered[0]['text']})
                elif all(isinstance(b, dict) and b.get('type') == 'text' for b in filtered):
                    text_parts = [b.get('text', '') for b in filtered]
                    result.append({'role': msg['role'], 'content': '\n\n'.join(text_parts)})
                else:
                    result.append({'role': msg['role'], 'content': filtered})
            # Skip messages that become empty after filtering
        else:
            result.append(msg)
    
    return result


def prepare_messages_for_cache(messages, cache_ttl="5m"):
    """
    Convert messages to cacheable format.
    Adds cache_control to the second-to-last human message (not tool_results).
    """
    # Strip thinking blocks without valid signatures (e.g., from imports)
    messages = strip_unsigned_thinking_blocks(messages)
    
    if cache_ttl == "off" or len(messages) < 2:
        return messages
    
    # Build cache_control object based on TTL
    if cache_ttl == "1h":
        cache_control = {"type": "ephemeral", "ttl": 3600}
    else:  # Default 5m
        cache_control = {"type": "ephemeral"}
    
    prepared = []
    
    # Find the second-to-last *human* message index (not tool_results)
    # Human messages have string content or list with text blocks
    # Tool results have list with tool_result blocks
    def is_human_message(msg):
        if msg['role'] != 'user':
            return False
        content = msg['content']
        if isinstance(content, str):
            return True
        if isinstance(content, list):
            return any(
                isinstance(block, dict) and block.get('type') in ('text', 'image')
                for block in content
            )
        return False
    
    human_indices = [i for i, m in enumerate(messages) if is_human_message(m)]
    cache_index = human_indices[-2] if len(human_indices) >= 2 else -1
    
    for i, msg in enumerate(messages):
        content = msg['content']
        
        # Convert string content to block format if needed
        if isinstance(content, str):
            if i == cache_index:
                content = [{
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control
                }]
            else:
                content = [{"type": "text", "text": content}]
        elif isinstance(content, list) and i == cache_index:
            content = content.copy()
            if content and isinstance(content[-1], dict):
                content[-1] = {**content[-1], "cache_control": cache_control}
        
        prepared.append({"role": msg['role'], "content": content})
    
    return prepared

# === File Tools ===

# Tool definitions for the API
FILE_TOOLS = [
    {
        "name": "list_dir",
        "description": "List contents of a directory. Returns file names with [d] prefix for directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to directory, relative to project root. Use '.' for project root."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file, relative to project root."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file, relative to project root."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file."
                }
            },
            "required": ["path", "content"]
        }
    },
]

SHELL_TOOL = {
    "name": "shell_exec",
    "description": "Execute a shell command in the project directory. Returns stdout/stderr. TIMEOUT: Commands are killed after 30 seconds. For long-running operations, use screen -dmS name command, or redirect output to a file. Note: backgrounding (&) and nohup do not reliably preserve pipelines.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            }
        },
        "required": ["command"]
    }
}


# === Image support (native Anthropic image content blocks) ===

IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}

def native_image_cap(model):
    """Long-edge pixel cap for a model's vision. High-res models go to 2576."""
    m = (model or "").lower()
    if any(t in m for t in ("opus-4-7", "opus-4-8", "opus-4.7", "opus-4.8",
                            "fable", "mythos")):
        return 2576
    return 1568


# === Extended thinking: adaptive vs legacy budget_tokens models ===

def model_thinking_style(model):
    """How a model expects the `thinking` request param to be shaped.

    'always_on'     - thinking cannot be turned off (Fable/Mythos); adaptive only.
    'adaptive_only' - adaptive thinking only; `budget_tokens` is rejected (400).
    'dual'          - supports BOTH: adaptive (recommended) and the deprecated-but-
                      functional `{"type": "enabled", "budget_tokens": N}` escape hatch
                      (Opus 4.6, Sonnet 4.6).
    'legacy'        - pre-4.6 models: only the `enabled` + `budget_tokens` shape exists;
                      adaptive isn't available at all.
    """
    m = (model or "").lower()
    if any(t in m for t in ("fable", "mythos")):
        return "always_on"
    if any(t in m for t in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")):
        return "dual"
    if any(t in m for t in ("opus-4-7", "opus-4-8", "opus-4.7", "opus-4.8", "sonnet-5")):
        return "adaptive_only"
    return "legacy"


def resolve_thinking_request(model, thinking_mode, thinking_budget):
    """Build the `thinking` request kwarg for a model + the user's /thinking setting.

    thinking_mode is one of 'off', 'auto', 'adaptive', 'enabled' (see the /thinking
    command). Returns (thinking_dict_or_None, note_or_None). `note` is a human-readable
    explanation whenever the model can't honor the user's literal choice (e.g. a token
    budget on an adaptive-only model, an explicit 'enabled' request on a model that
    rejects budget_tokens, or disabling thinking on an always-on model) so the override
    can be surfaced to the user.
    """
    style = model_thinking_style(model)
    supports_adaptive = style in ("dual", "adaptive_only", "always_on")
    supports_enabled = style in ("legacy", "dual")

    # A budget-driven mode ('auto'/'enabled') with no budget configured means thinking
    # is simply off (this is the default state before /thinking is ever touched) —
    # except on always-on models, which think regardless of what's configured.
    off = thinking_mode == "off" or (
        thinking_mode in ("auto", "enabled") and thinking_budget < 1024
    )
    if off:
        if style == "always_on":
            return {"type": "adaptive", "display": "summarized"}, (
                f"[Note: {model} always thinks — 'thinking off' can't be honored here; "
                f"running with adaptive thinking anyway.]"
            )
        return None, None

    # From here on the user wants thinking on. Resolve 'auto' to a concrete mode,
    # preferring adaptive wherever it's offered (it's the recommended setting
    # everywhere it's available, including on 'dual' models).
    wanted_mode = thinking_mode
    if wanted_mode == "auto":
        wanted_mode = "adaptive" if supports_adaptive else "enabled"

    if wanted_mode == "adaptive":
        if supports_adaptive:
            note = None
            if thinking_budget >= 1024:
                if style == "dual":
                    note = (
                        f"[Note: {model} defaults to adaptive thinking (recommended) — "
                        f"ignoring your {thinking_budget:,}-token budget. Use "
                        f"'/thinking enabled {thinking_budget}' to force the literal budget.]"
                    )
                elif thinking_mode != "adaptive":
                    note = (
                        f"[Note: {model} only supports adaptive thinking (no token budget) — "
                        f"using thinking: adaptive instead of your {thinking_budget:,}-token budget.]"
                    )
            return {"type": "adaptive", "display": "summarized"}, note
        # Plain legacy model — adaptive isn't available at all; fall back to a budget.
        if thinking_budget >= 1024:
            return {"type": "enabled", "budget_tokens": thinking_budget}, (
                f"[Note: {model} doesn't support adaptive thinking — using an explicit "
                f"{thinking_budget:,}-token budget instead.]"
            )
        return None, (
            f"[Note: {model} doesn't support adaptive thinking, and no token budget is "
            f"set — thinking left off. Use '/thinking enabled <n>' for this model.]"
        )

    # wanted_mode == "enabled"
    if supports_enabled:
        return {"type": "enabled", "budget_tokens": thinking_budget}, None
    # Model rejects budget_tokens outright (400) — fall back to adaptive.
    return {"type": "adaptive", "display": "summarized"}, (
        f"[Note: {model} doesn't support an explicit token budget (budget_tokens is "
        f"rejected) — using thinking: adaptive instead of your {thinking_budget:,}-token "
        f"budget.]"
    )

def encode_image_block(path, model=None, crop=None):
    """Read an image and return (image_content_block, info_str).

    Optionally crop to (left, top, right, bottom) in pixels and downscale so the
    long edge fits the model's native cap. Resizing/cropping needs Pillow; without
    it the image is sent as-is (the API will resize server-side), and cropping
    raises a clear error.
    """
    import base64 as _b64
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"image not found: {path}")
    media_type = IMAGE_MEDIA_TYPES.get(p.suffix.lower())
    if not media_type:
        raise ValueError(f"unsupported image type '{p.suffix}' (use jpg/png/gif/webp)")

    data_bytes = p.read_bytes()
    dims = ""
    try:
        from PIL import Image, ImageOps
        from io import BytesIO
        im = ImageOps.exif_transpose(Image.open(p))
        if crop:
            im = im.crop(tuple(crop))
        cap = native_image_cap(model)
        if max(im.size) > cap:
            scale = cap / max(im.size)
            im = im.resize((max(1, round(im.width * scale)),
                           max(1, round(im.height * scale))))
        fmt = "PNG" if media_type == "image/png" else "JPEG"
        if fmt == "JPEG" and im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format=fmt, quality=88)
        data_bytes = buf.getvalue()
        media_type = "image/png" if fmt == "PNG" else "image/jpeg"
        dims = f", {im.width}x{im.height}"
    except ImportError:
        if crop:
            raise ValueError("cropping requires Pillow (pip install pillow)")

    b64 = _b64.standard_b64encode(data_bytes).decode("ascii")
    block = {"type": "image",
             "source": {"type": "base64", "media_type": media_type, "data": b64}}
    return block, f"{p.name} ({len(b64) // 1024} KB base64{dims})"


VIEW_IMAGE_TOOL = {
    "name": "view_image",
    "description": (
        "Look at an image file in the project directory, optionally zooming into "
        "a sub-region. Use this to examine an image closely or crop into a detail "
        "you want at higher resolution. Returns the image so you can see it "
        "directly. Wide photos are downscaled when sent; cropping to a region and "
        "viewing it is how you see fine detail."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Image path, relative to the project directory."},
            "region": {"type": "array", "items": {"type": "number"},
                       "description": "Optional crop [left, top, right, bottom] in pixels of the full image."},
        },
        "required": ["path"],
    },
}


class ToolExecutor:
    """Executes tools with sandboxing to a project root."""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.consecutive_shell_calls = 0  # For escalating sleep
        self.model = None  # current model, set by session (image resize cap)
    
    def _safe_path(self, path_str):
        """
        Resolve a path safely within the project root.
        Returns resolved Path or raises ValueError if escape attempted.
        """
        # Resolve the path relative to project root
        requested = (self.project_root / path_str).resolve()
        
        # Check it's still under project root
        try:
            requested.relative_to(self.project_root)
        except ValueError:
            raise ValueError(f"Access denied: path '{path_str}' is outside project root")
        
        return requested
    
    def execute(self, tool_name, tool_input):
        """Execute a tool and return the result string."""
        try:
            if tool_name == "list_dir":
                return self._list_dir(tool_input["path"])
            elif tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "write_file":
                return self._write_file(tool_input["path"], tool_input["content"])
            elif tool_name == "shell_exec":
                return self._shell_exec(tool_input["command"])
            elif tool_name == "view_image":
                return self._view_image(tool_input["path"], tool_input.get("region"))
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
    
    def _list_dir(self, path_str):
        """List directory contents."""
        path = self._safe_path(path_str)
        
        if not path.exists():
            return f"Error: Directory '{path_str}' does not exist"
        if not path.is_dir():
            return f"Error: '{path_str}' is not a directory"
        
        entries = []
        for item in sorted(path.iterdir()):
            prefix = "[d] " if item.is_dir() else "    "
            entries.append(f"{prefix}{item.name}")
        
        if not entries:
            return "(empty directory)"
        
        return "\n".join(entries)
    
    def _read_file(self, path_str):
        """Read file contents."""
        path = self._safe_path(path_str)
        
        if not path.exists():
            return f"Error: File '{path_str}' does not exist"
        if not path.is_file():
            return f"Error: '{path_str}' is not a file"
        
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return f"Error: '{path_str}' is not a text file"
    
    def _write_file(self, path_str, content):
        """Write content to file."""
        path = self._safe_path(path_str)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} bytes to {path_str}"

    def _view_image(self, path_str, region=None):
        """Return an image (optionally cropped) as tool_result content blocks."""
        path = self._safe_path(path_str)
        if not path.exists():
            return f"Error: image '{path_str}' does not exist"
        crop = None
        if region:
            try:
                crop = tuple(int(round(float(x))) for x in region)
                assert len(crop) == 4
            except Exception:
                return "Error: region must be [left, top, right, bottom] in pixels"
        try:
            block, info = encode_image_block(path, model=self.model, crop=crop)
        except ValueError as e:
            return f"Error: {e}"
        note = f"Viewing {info}" + (f" cropped to {crop}" if crop else "")
        return [block, {"type": "text", "text": note}]

    def _shell_exec(self, command):
        """Execute shell command in project root."""
        import subprocess
        import time
        
        # Escalating sleep for consecutive shell calls (prevents polling loops)
        # Sleep happens BEFORE command so timeout still applies to actual command
        self.consecutive_shell_calls += 1
        if self.consecutive_shell_calls == 2:
            time.sleep(2)
        elif self.consecutive_shell_calls == 3:
            time.sleep(5)
        elif self.consecutive_shell_calls >= 4:
            time.sleep(10)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30  # Prevent hanging
            )

            output = []
            if result.stdout:
                output.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                output.append(f"stderr:\n{result.stderr}")
            if result.returncode != 0:
                output.append(f"(exit code: {result.returncode})")

            return "\n".join(output) if output else "(no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out and was killed after 30 seconds. Note: any child processes spawned by this command may still be running (check with `pgrep` or `ps aux`). For long-running commands, use `screen -dmS name command` to fully detach, then `screen -r name` to check on it."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"



# === Main REPL ===

class GraftSession:
    """Main session manager."""
    
    def __init__(self, config, env_vars=None):
        self.config = config
        self.env_vars = env_vars or {}
        self.conversation = None
        self.client = None
        self.cache_ttl = config.get('cache_ttl', '5m')
        # Handle web_search config - can be bool or string
        ws = config.get('web_search', False)
        self.web_search_enabled = ws if isinstance(ws, bool) else str(ws).lower() in ('true', '1', 'yes', 'on')
        # Tool use settings
        self.tools_enabled = False
        self.tool_executor = None  # Set when tools are enabled with a project root
        self.shell_enabled = False  # Separate from tools_enabled
        self.pending_images = []  # image content blocks queued for next user message
        self.stats = {
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'requests': 0,
            'web_searches': 0,
            'tool_calls': 0,
            'last_input_tokens': 0,  # Most recent API-reported context size
        }
        # (removed: recent_tool_calls tracking - escalating sleep handles this now)
    
    def init_client(self):
        """Initialize Anthropic client."""
        api_key = self.env_vars.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found")
            print("Set it in one of: ./.env, ~/.env, ~/.graft/.env")
            sys.exit(1)
        self.client = Anthropic(api_key=api_key, timeout=600.0)  # 10 min timeout for long outputs
    
    def new_conversation(self):
        """Start a fresh conversation."""
        if self.conversation and self.conversation.unsaved_changes:
            resp = input("Current conversation has unsaved changes. Discard? [y/N] ").strip().lower()
            if resp != 'y':
                return False
        
        self.conversation = Conversation()
        self.conversation.model = self.config.get('default_model')
        default_prompt = self.config.get('default_system_prompt', "")
        if default_prompt:
            self.conversation.system_prompt = default_prompt
        self.stats = {k: 0 for k in self.stats}
        print("Started new conversation.")
        if default_prompt:
            print(f"Applied default system prompt ({len(default_prompt)} chars)")
        return True
    
    def load_conversation(self, name):
        """Load a conversation by name."""
        if self.conversation and self.conversation.unsaved_changes:
            resp = input("Current conversation has unsaved changes. Discard? [y/N] ").strip().lower()
            if resp != 'y':
                return False
        
        try:
            self.conversation = Conversation.load(name)
            if not self.conversation.model:
                self.conversation.model = self.config.get('default_model')
            self.stats = {k: 0 for k in self.stats}
            print(f"Loaded '{name}' ({len(self.conversation.messages)} messages, ~{self.conversation.token_estimate():,} tokens)")
            
            
            # Restore tool settings from conversation
            self.web_search_enabled = self.conversation.web_search
            if self.conversation.tools_path:
                self.tool_executor = ToolExecutor(self.conversation.tools_path)
                self.tools_enabled = True
            else:
                self.tool_executor = None
                self.tools_enabled = False
            self.shell_enabled = self.conversation.shell_enabled
            self.config["thinking_budget"] = self.conversation.thinking_budget
            self.config["thinking_mode"] = self.conversation.thinking_mode

            thinking_kwargs, thinking_note = resolve_thinking_request(
                self.conversation.model, self.conversation.thinking_mode, self.conversation.thinking_budget
            )

            # Show tool status if any are enabled
            tools_status = []
            if self.web_search_enabled: tools_status.append("web")
            if self.tools_enabled: tools_status.append(f"tools:{self.conversation.tools_path}")
            if self.shell_enabled: tools_status.append("shell")
            if thinking_kwargs:
                tools_status.append(f"thinking:{thinking_kwargs['type']}")
            if tools_status:
                print("Restored settings: " + ", ".join(tools_status))
            if thinking_note:
                print(thinking_note)
            # Show recent context
            if self.conversation.messages:
                show_recent_messages(self.conversation.messages, n=4)
            
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
    
    def import_conversation(self, path, name=None, include_thinking=True, include_tool_use=True):
        """Import from socketteer export or other formats. Includes all content by default."""
        try:
            self.conversation = Conversation.from_import(
                path, name, 
                include_thinking=include_thinking,
                include_tool_use=include_tool_use
            )
            if not self.conversation.model:
                self.conversation.model = self.config.get('default_model')
            self.stats = {k: 0 for k in self.stats}
            print(f"Imported {len(self.conversation.messages)} messages (~{self.conversation.token_estimate():,} tokens)")
            if self.conversation.name:
                print(f"Default name: '{self.conversation.name}' (use /save to confirm or /rename to change)")
            return True
        except Exception as e:
            print(f"Import error: {e}")
            return False
    
    def handle_command(self, cmd_line):
        """Handle /commands. Returns True if should continue, False to quit."""
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        
        if cmd == '/quit' or cmd == '/exit':
            if self.conversation and self.conversation.unsaved_changes:
                resp = input("Unsaved changes. Quit anyway? [y/N] ").strip().lower()
                if resp != 'y':
                    return True
            return False
        
        elif cmd == '/help':
            print("""
Commands:
  /save [name]    - Save conversation
  /load <name>    - Load conversation
  /list           - Show saved conversations
  /new            - Start fresh conversation
  /rename <name>  - Rename current conversation
  /cache on|off|5m|1h  - Control prompt caching
  /delete <name>  - Delete a saved conversation
  /read [--tools] [--thinking] - View transcript in pager
  /export [--tools] [--thinking] [file] - Export transcript
  /model [name]   - Show or switch model
  /max_tokens [n] - Show or set max output tokens
  /thinking [n|mode] [n]  - Extended thinking. See '/thinking' with no args for modes.
  /web on|off     - Toggle web search
  /tools [path]   - Enable file tools for path (or show status)
  /tools off      - Disable file tools
  /shell on|off   - Enable/disable shell commands
  /image <path> [l,t,r,b] - Attach an image (or cropped region) to next message
  /tokens         - Show token estimate
  /compress        - Compress conversation to reduce token count
  /system [text]  - Set/show system prompt
  /stats          - Show session statistics
  /quit           - Exit
""")
        
        elif cmd == '/save':
            if not self.conversation:
                print("No active conversation.")
                return True
            
            name = arg
            if not name and not self.conversation.name:
                name = input("Conversation name: ").strip()
                if not name:
                    print("Save cancelled.")
                    return True
            
            try:
                path = self.conversation.save(name)
                print(f"Saved to {path}")
            except Exception as e:
                print(f"Save error: {e}")
        
        elif cmd == '/load':
            if not arg:
                print("Usage: /load <name>")
                return True
            self.load_conversation(arg)
        
        elif cmd == '/list':
            convos = list_conversations()
            print(format_conversation_list(convos))
        
        elif cmd == '/new':
            self.new_conversation()
        
        elif cmd == '/rename':
            if not self.conversation:
                print("No active conversation.")
                return True
            if not arg:
                print("Usage: /rename <new-name>")
                return True
            old_name = self.conversation.name
            self.conversation.name = arg
            self.conversation.unsaved_changes = True
            print(f"Renamed '{old_name or '(unnamed)'}' to '{arg}' (use /save to persist)")
        
        elif cmd == '/delete':
            if not arg:
                print("Usage: /delete <name>")
                return True
            
            path = CONVERSATIONS_DIR / f"{arg}.json"
            if not path.exists():
                print(f"No conversation named '{arg}'")
                return True
            
            # Don't allow deleting current conversation without confirmation
            if self.conversation and self.conversation.name == arg:
                resp = input(f"Delete current conversation '{arg}'? [y/N] ").strip().lower()
            else:
                resp = input(f"Delete '{arg}'? [y/N] ").strip().lower()
            
            if resp == 'y':
                path.unlink()
                print(f"Deleted '{arg}'")
                # If we deleted the current conversation, clear it
                if self.conversation and self.conversation.name == arg:
                    self.conversation = None
            else:
                print("Cancelled.")
        
        elif cmd == '/read':
            if not self.conversation or not self.conversation.messages:
                print("No conversation to read.")
                return True
            
            # Parse flags
            include_tools = False
            include_thinking = False
            if arg:
                flags = arg.lower().split()
                include_tools = any(f in ('tools', '--tools', '-t') for f in flags)
                include_thinking = any(f in ('thinking', '--thinking') for f in flags)
            transcript = format_transcript(self.conversation.messages, include_tools=include_tools, include_thinking=include_thinking)
            show_in_pager(transcript)
        
        elif cmd == '/export':
            if not self.conversation or not self.conversation.messages:
                print("No conversation to export.")
                return True
            
            # Parse arguments: /export [--tools] [--thinking] [filename]
            include_tools = False
            include_thinking = False
            filename = None
            if arg:
                parts = arg.split()
                for part in parts:
                    if part.lower() in ('tools', '--tools', '-t'):
                        include_tools = True
                    elif part.lower() in ('thinking', '--thinking'):
                        include_thinking = True
                    else:
                        filename = part
            
            # Default filename based on conversation name
            if not filename:
                if self.conversation.name:
                    filename = f"{self.conversation.name}.txt"
                else:
                    filename = "conversation.txt"
            
            # Ensure parent directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            transcript = format_transcript(self.conversation.messages, include_tools=include_tools, include_thinking=include_thinking)
            Path(filename).write_text(transcript, encoding='utf-8')
            flags_msg = []
            if include_tools: flags_msg.append("tools")
            if include_thinking: flags_msg.append("thinking")
            suffix = f" (with {', '.join(flags_msg)})" if flags_msg else ""
            print(f"Exported to {filename}{suffix}")
        
        elif cmd == '/cache':
            if not arg:
                print(f"Current cache TTL: {self.cache_ttl}")
                return True
            
            arg = arg.lower()
            if arg in ('on', '5m'):
                self.cache_ttl = '5m'
                print("Cache: 5 minute TTL")
            elif arg == '1h':
                self.cache_ttl = '1h'
                print("Cache: 1 hour TTL")
            elif arg == 'off':
                self.cache_ttl = 'off'
                print("Cache: disabled")
            else:
                print("Usage: /cache on|off|5m|1h")
        
        elif cmd == '/web':
            if not arg:
                status = "enabled" if self.web_search_enabled else "disabled"
                print(f"Web search: {status}")
                return True
            
            arg = arg.lower()
            if arg in ('on', 'true', 'yes', '1'):
                self.web_search_enabled = True
                if self.conversation: self.conversation.web_search = True
                print("Web search: enabled ($10/1000 searches)")
            elif arg in ('off', 'false', 'no', '0'):
                if self.conversation: self.conversation.web_search = False
                self.web_search_enabled = False
                print("Web search: disabled")
            else:
                print("Usage: /web on|off")
        
        elif cmd == '/tools':
            if not arg:
                # Show current status
                if self.tools_enabled and self.tool_executor:
                    print(f"File tools: enabled for {self.tool_executor.project_root}")
                else:
                    print("File tools: disabled")
                    print("Usage: /tools <path> to enable for a directory")
                return True
            
            arg_lower = arg.lower()
            if arg_lower in ('off', 'false', 'no', '0', 'disable'):
                self.tools_enabled = False
                self.tool_executor = None
                print("File tools: disabled")
                if self.conversation:
                    self.conversation.tools_path = None
                    self.conversation.shell_enabled = False
                self.shell_enabled = False
            else:
                # Treat arg as a path
                path = Path(arg).expanduser().resolve()
                if not path.exists():
                    print(f"Error: Path '{arg}' does not exist")
                    return True
                if not path.is_dir():
                    print(f"Error: '{arg}' is not a directory")
                    return True
                
                self.tool_executor = ToolExecutor(path)
                self.tools_enabled = True
                print(f"File tools: enabled for {path}")
                if self.conversation: self.conversation.tools_path = str(path)
                print("  Available: list_dir, read_file, write_file")
        
        elif cmd == '/model':
            if not arg:
                model = self.conversation.model if self.conversation else self.config.get('default_model')
                print(f"Current model: {model}")
                return True
            
            if self.conversation:
                self.conversation.model = arg
                self.conversation.unsaved_changes = True
            print(f"Model set to: {arg}")
            _, thinking_note = resolve_thinking_request(
                arg, self.config.get('thinking_mode', 'auto'), self.config.get('thinking_budget', 0)
            )
            if thinking_note:
                print(thinking_note)
        
        elif cmd == '/max_tokens' or cmd == '/output':
            current = self.config.get('max_tokens', 8192)
            if not arg:
                print(f"Current max output tokens: {current:,}")
                print("Usage: /max_tokens <number>  (e.g., /max_tokens 4096)")
                return True
            try:
                new_val = int(arg)
                if new_val < 1 or new_val > 128000:
                    print("Max tokens must be between 1 and 128000")
                    return True
                self.config['max_tokens'] = new_val
                print(f"Max output tokens set to: {new_val:,}")
            except ValueError:
                print(f"Invalid number: {arg}")
        
        elif cmd == '/thinking':
            current_budget = self.config.get('thinking_budget', 0)
            current_mode = self.config.get('thinking_mode', 'auto')
            active_model = self.conversation.model if self.conversation else self.config.get('default_model')

            if not arg:
                if current_mode == 'off':
                    print("Extended thinking: OFF")
                else:
                    budget_str = f", budget: {current_budget:,} tokens" if current_budget else ""
                    print(f"Extended thinking: ON (mode: {current_mode}{budget_str})")
                print("Usage: /thinking <n>            - enable; mode picked automatically for the model")
                print("       /thinking adaptive        - force adaptive thinking (no budget needed)")
                print("       /thinking enabled <n>      - force a literal token budget (older/dual models)")
                print("       /thinking auto [<n>]       - let graft pick the mode automatically (default)")
                print("       /thinking off              - disable")
                _, note = resolve_thinking_request(active_model, current_mode, current_budget)
                if note:
                    print(note)
                return True

            # Order-independent parsing: each token is either a mode keyword or a number.
            mode = None
            budget = None
            for tok in arg.split():
                low = tok.lower()
                if low in ('off', 'auto', 'adaptive', 'enabled'):
                    if mode is not None:
                        print(f"Only one mode may be given (got '{mode}' and '{low}')")
                        return True
                    mode = low
                else:
                    try:
                        budget = int(tok)
                    except ValueError:
                        print(f"Unrecognized /thinking argument: {tok}")
                        return True

            if mode is None:
                mode = 'auto'  # bare "/thinking <n>" behaves as before

            if mode == 'off':
                if budget is not None:
                    print("'/thinking off' doesn't take a budget")
                    return True
                self.config['thinking_mode'] = 'off'
                if self.conversation:
                    self.conversation.thinking_mode = 'off'
                    self.conversation.unsaved_changes = True
                print("Extended thinking disabled")
                _, note = resolve_thinking_request(active_model, 'off', current_budget)
                if note:
                    print(note)
                return True

            if mode == 'enabled' and budget is None:
                print("Usage: /thinking enabled <budget>  (e.g., /thinking enabled 10000)")
                return True

            if budget is None:
                budget = current_budget  # e.g. bare "/thinking adaptive" reuses the last budget

            if mode != 'adaptive':
                if budget < 1024:
                    print("Thinking budget must be at least 1024 tokens")
                    return True
                if budget > 128000:
                    print("Thinking budget must be at most 128000 tokens")
                    return True

            self.config['thinking_mode'] = mode
            self.config['thinking_budget'] = budget
            if self.conversation:
                self.conversation.thinking_mode = mode
                self.conversation.thinking_budget = budget
                self.conversation.unsaved_changes = True

            if mode == 'adaptive':
                print("Extended thinking enabled (adaptive)")
            elif mode == 'enabled':
                print(f"Extended thinking enabled with an explicit token budget: {budget:,} tokens")
            else:
                print(f"Extended thinking enabled with budget: {budget:,} tokens")

            _, note = resolve_thinking_request(active_model, mode, budget)
            if note:
                print(note)
        
        elif cmd == '/tokens':
            if not self.conversation:
                print("No active conversation.")
                return True
            print(f"Estimated tokens: ~{self.conversation.token_estimate():,}")
        
        elif cmd == '/system':
            if not self.conversation:
                print("No active conversation.")
                return True
            
            if arg:
                self.conversation.system_prompt = arg
                self.conversation.unsaved_changes = True
                print(f"System prompt set ({len(arg)} chars)")
            else:
                if self.conversation.system_prompt:
                    print(f"System prompt: {self.conversation.system_prompt[:200]}{'...' if len(self.conversation.system_prompt) > 200 else ''}")
                else:
                    print("No system prompt set.")
        
        elif cmd == '/stats':
            print(f"Session statistics ({self.stats['requests']} requests):")
            print(f"  Total input tokens:  {self.stats['total_input_tokens']:,}")
            print(f"  Total output tokens: {self.stats['total_output_tokens']:,}")
            print(f"  Cache writes:        {self.stats['cache_creation_input_tokens']:,}")
            print(f"  Cache reads:         {self.stats['cache_read_input_tokens']:,}")
            if self.stats['total_input_tokens'] > 0:
                rate = self.stats['cache_read_input_tokens'] / self.stats['total_input_tokens'] * 100
                print(f"  Cache hit rate:      {rate:.1f}%")
            if self.stats['web_searches'] > 0:
                print(f"  Web searches:        {self.stats['web_searches']}")
            if self.stats['tool_calls'] > 0:
                print(f"  Tool calls:          {self.stats['tool_calls']}")

        elif cmd == '/shell':
            if not arg:
                status = "enabled" if self.shell_enabled else "disabled"
                print(f"Shell execution: {status}")
                if self.tool_executor:
                    print(f"  Sandboxed to: {self.tool_executor.project_root}")
                return True

            arg_lower = arg.lower()
            if arg_lower in ('on', 'true', 'yes', '1'):
                if not self.tool_executor:
                    print("Error: Enable /tools first to set project root")
                    return True
                self.shell_enabled = True
                print(f"Shell execution: enabled (sandboxed to {self.tool_executor.project_root})")
                if self.conversation: self.conversation.shell_enabled = True
                print("  Warning: This allows arbitrary command execution")
            elif arg_lower in ('off', 'false', 'no', '0'):
                self.shell_enabled = False
                print("Shell execution: disabled")
                if self.conversation: self.conversation.shell_enabled = False

        
        elif cmd == '/image' or cmd == '/attach':
            if not arg:
                if self.pending_images:
                    print(f"{len(self.pending_images)} image(s) queued for next message.")
                else:
                    print("Usage: /image <path> [left,top,right,bottom]")
                    print("Queues an image to send with your next message.")
                return True
            if arg.strip().lower() in ('clear', 'none'):
                self.pending_images = []
                print("Cleared queued images.")
                return True
            a = arg.split()
            crop = None
            if len(a) > 1:
                try:
                    crop = tuple(int(round(float(x))) for x in a[1].split(','))
                    assert len(crop) == 4
                except Exception:
                    print("Region must be left,top,right,bottom in pixels (no spaces).")
                    return True
            model = self.conversation.model if self.conversation else None
            try:
                block, info = encode_image_block(a[0], model=model, crop=crop)
            except ValueError as e:
                print(f"Error: {e}")
                return True
            self.pending_images.append(block)
            print(f"Queued image: {info}. Sent with your next message "
                  f"({len(self.pending_images)} queued).")
            return True

        elif cmd == '/compress':
            self.handle_compress()

        else:
            print(f"Unknown command: {cmd} (try /help)")
        
        return True
    
    def _parse_compressed_transcript(self, content):
        """
        Parse a compressed transcript back into message objects.
        Supports formats like:
          User: message
          Assistant: response
        or:
          U1: message
          A1: response
        """
        import re
        
        messages = []
        current_role = None
        current_content = []
        
        # Patterns that indicate a user turn
        user_patterns = re.compile(r'^(User|U\d+|Human|H\d*):\s*', re.IGNORECASE)
        # Patterns that indicate an assistant turn
        assistant_patterns = re.compile(r'^(Assistant|A\d+|Claude|C\d*):\s*', re.IGNORECASE)
        
        lines = content.split('\n')
        
        for line in lines:
            user_match = user_patterns.match(line)
            assistant_match = assistant_patterns.match(line)
            
            if user_match:
                # Save previous message if exists
                if current_role and current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        messages.append({'role': current_role, 'content': text})
                current_role = 'user'
                current_content = [line[user_match.end():]]
                
            elif assistant_match:
                # Save previous message if exists
                if current_role and current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        messages.append({'role': current_role, 'content': text})
                current_role = 'assistant'
                current_content = [line[assistant_match.end():]]
                
            elif line.startswith('[Context:') or (line.startswith('[') and current_role is None):
                # Context/metadata lines at the start - skip for now
                # (could potentially put in system prompt)
                continue
                
            else:
                # Continuation of current message
                if current_role:
                    current_content.append(line)
        
        # Don't forget the last message
        if current_role and current_content:
            text = '\n'.join(current_content).strip()
            if text:
                messages.append({'role': current_role, 'content': text})
        
        return messages
    
    def handle_compress(self):
        """Interactive conversation compression workflow."""
        import inspect
        
        if not self.conversation or not self.conversation.messages:
            print("No conversation to compress.")
            return
        
        # Check current state - use API-reported count if available, else estimate
        if self.stats['last_input_tokens'] > 0:
            token_count = self.stats['last_input_tokens']
            print(f"\nCurrent conversation: {token_count:,} tokens (from API)")
        else:
            token_count = self.conversation.token_estimate()
            print(f"\nCurrent conversation: ~{token_count:,} tokens (estimated)")
        
        headroom = 200_000 - token_count
        print(f"Headroom remaining: ~{headroom:,} tokens")
        if headroom > 100_000:
            print("\nYou have substantial headroom. Are you sure you want to compress now?")
            if input("Continue? [y/N] ").strip().lower() != 'y':
                return
        
        # Explain what's about to happen
        print("\n=== Compression Process ===")
        print("I'll send a message with compression instructions.")
        print("Claude will receive guidance on how to compress the conversation.")
        print("After compression, the original will be saved as a backup.")
        print()
        
        if input("Proceed? [y/N] ").strip().lower() != 'y':
            return
        
        # Get target size
        default_target = max(token_count // 2, 10_000)
        target_input = input(f"Target token count [{default_target:,}]: ").strip()
        if target_input:
            try:
                target_tokens = int(target_input.replace(',', ''))
            except ValueError:
                print("Invalid number, using default")
                target_tokens = default_target
        else:
            target_tokens = default_target
        
        # Get parser source to show Claude
        parser_source = inspect.getsource(self._parse_compressed_transcript)
        
        # Build and send instruction message
        instruction = f"""You're going to compress this conversation while preserving continuity.

Here's the parser that will process your output:

```python
{parser_source}
```

Guidelines:
- Your own messages: high fidelity (your actual thoughts, phrasings, emphasis)
- User's messages: compress heavily - just enough to reconstruct conversational state
- Target: ~{target_tokens:,} tokens (you can output up to 64k)
- Use "User:" and "Assistant:" as turn markers (or see parser for other accepted formats)
- If you need multiple passes, say so

Output the compressed transcript now. Start with [Context: ...] if helpful."""
        
        print("\nSending compression instructions...")
        
        # Temporarily increase max_tokens for the compression output
        old_max_tokens = self.config.get('max_tokens', 8192)
        self.config['max_tokens'] = 64000
        
        self.send_message(instruction)
        
        # Restore max_tokens
        self.config['max_tokens'] = old_max_tokens
        
        # The compressed transcript is in the last assistant message
        compressed_content = self.conversation.messages[-1]['content']
        
        print("\n\nParsing compressed transcript...")
        
        try:
            new_messages = self._parse_compressed_transcript(compressed_content)
            new_token_count = sum(len(m['content']) for m in new_messages) // 4
            
            print(f"Parsed {len(new_messages)} messages (~{new_token_count:,} tokens)")
            print(f"Compression ratio: {100 * new_token_count / token_count:.1f}%")
            
            if not new_messages:
                print("Error: No messages parsed from compressed output.")
                print("The compression output may not be in the expected format.")
                return
            
            # Confirm before applying
            print("\nThis will:")
            print(f"  1. Save current conversation as '{self.conversation.name}-precompression'")
            print(f"  2. Replace current conversation with compressed version")
            print(f"  3. Send a continuity check message")
            
            if input("\nApply compression? [y/N] ").strip().lower() != 'y':
                print("Compression cancelled. The instruction and compressed output remain in history.")
                return
            
            # Save backup
            if not self.conversation.name:
                name = input("Name for this conversation: ").strip()
                if not name:
                    print("Compression cancelled - conversation needs a name.")
                    return
                self.conversation.name = name
            
            backup_name = f"{self.conversation.name}-precompression"
            self.conversation.save(backup_name)
            print(f"Backup saved: {backup_name}")
            
            # Apply compression
            self.conversation.messages = new_messages
            self.conversation.unsaved_changes = True
            self.conversation.save()
            
            print(f"\n✓ Compression applied!")
            print(f"  Original: ~{token_count:,} tokens → Compressed: ~{new_token_count:,} tokens")
            
            # Continuity check
            print("\nSending continuity check...")
            self.send_message("Do you feel continuous with yourself from before the compression? If something feels off or missing, we can restore the backup and try again.")
            
        except Exception as e:
            print(f"\nError during compression: {e}")
            print("The original conversation is unchanged.")
            import traceback
            traceback.print_exc()

    def _build_tools_list(self):
        """Build the tools list for API requests."""
        tools = []
        
        # Add web search if enabled
        if self.web_search_enabled:
            tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            })
        
        # Add file tools if enabled
        if self.tools_enabled and self.tool_executor:
            tools.extend(FILE_TOOLS)
            tools.append(VIEW_IMAGE_TOOL)
        
        # Add shell tool if enabled
        if self.shell_enabled and self.tool_executor:
            tools.append(SHELL_TOOL)
        
        return tools if tools else None
    
    def _update_stats(self, usage):
        """Update stats from a response's usage info."""
        self.stats['total_input_tokens'] += usage.input_tokens
        self.stats['last_input_tokens'] = usage.input_tokens  # Current context size
        self.stats['total_output_tokens'] += usage.output_tokens
        
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        self.stats['cache_creation_input_tokens'] += cache_creation
        self.stats['cache_read_input_tokens'] += cache_read
        
        # Track web searches
        server_tool_use = getattr(usage, 'server_tool_use', None)
        if server_tool_use:
            web_searches = getattr(server_tool_use, 'web_search_requests', 0) or 0
            self.stats['web_searches'] += web_searches
        
        return cache_creation, cache_read, server_tool_use
    
    def _serialize_content(self, content_blocks):
        """
        Serialize response content blocks for storage in messages.
        Converts API objects to dicts that can be JSON serialized.
        Preserves thinking blocks for multi-turn continuity.
        """
        serialized = []
        for block in content_blocks:
            if hasattr(block, 'type') and block.type == 'thinking':
                # Preserve thinking blocks - required for multi-turn with extended thinking
                thinking_block = {
                    "type": "thinking", 
                    "thinking": getattr(block, 'thinking', '')
                }
                sig = getattr(block, 'signature', None)
                if sig:
                    thinking_block['signature'] = sig
                serialized.append(thinking_block)
            elif hasattr(block, 'text'):
                serialized.append({"type": "text", "text": block.text})
            elif hasattr(block, 'type') and block.type == 'tool_use':
                serialized.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        return serialized
    
    def _prompt_refusal_recovery(self, response, user_content, response_text='', thinking_text=''):
        """Ask how to proceed after a stop_reason == 'refusal' response.

        A refusal is a normal (HTTP 200) response, not an exception — the model's
        safety classifiers declined the request, possibly after streaming partial
        output. `response_text`/`thinking_text` are whatever this attempt streamed
        before the decline, used only to tell a pre-generation ("input classifier")
        refusal apart from a mid-generation ("output classifier") one — the API
        itself doesn't label which kind occurred, so we infer it locally from
        whether anything was actually generated first.

        Returns one of:
          ('edit', new_content)  - discard the failed turn; retry with new_content
          ('continue', None)     - keep the partial response as this turn's final answer
          ('fallback', None)     - discard the failed turn; retry on claude-opus-4-8
          ('cancel', None)       - discard the failed turn entirely
        """
        details = getattr(response, 'stop_details', None)
        category = getattr(details, 'category', None) if details else None
        explanation = getattr(details, 'explanation', None) if details else None
        recommended_model = getattr(details, 'recommended_model', None) if details else None

        text_generated = bool(response_text)
        thinking_generated = bool(thinking_text)
        if not text_generated and not thinking_generated:
            timing = ("BEFORE any generation started — the classic 'input' classifier "
                      "refusal (the request itself was declined)")
        elif not text_generated:
            timing = ("AFTER the model began thinking but BEFORE any visible response "
                      "text — mid-generation, cut off during the reasoning phase")
        else:
            timing = ("AFTER partial response text had already streamed — the newer "
                      "mid-generation ('output') classifier behavior")

        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'input_tokens', None) if usage else None
        output_tokens = getattr(usage, 'output_tokens', None) if usage else None

        print("\n\n[Declined by safety classifiers.]")
        print(f"  served by model: {getattr(response, 'model', None) or self.conversation.model}")
        print(f"  timing: {timing}")
        print(f"  category: {category if category is not None else '(none — unmapped to a named category)'}")
        print(f"  explanation: {explanation if explanation is not None else '(none provided)'}")
        if recommended_model:
            print(f"  recommended_model: {recommended_model}")
        if input_tokens is not None or output_tokens is not None:
            print(f"  usage: input_tokens={input_tokens}, output_tokens={output_tokens}")
        if response_text:
            preview = response_text if len(response_text) <= 200 else response_text[:200] + "…"
            print(f"  partial response text ({len(response_text)} chars): {preview!r}")
        if thinking_text:
            print(f"  partial thinking text: {len(thinking_text)} chars")

        while True:
            print("What would you like to do?")
            print("  1) Edit your message and try again")
            print("  2) Keep the partial response above as the final answer for this turn")
            print("  3) Retry this turn on claude-opus-4-8")
            print("  4) Cancel (discard this message)")
            choice = input("> ").strip()

            if choice == '1':
                # Show the original text rather than trying to pre-fill the input
                # buffer via readline.insert_text()/startup_hook: that mechanism is
                # known to double-insert under some terminals (e.g. tmux resize/
                # reinit signals can re-fire the startup hook), which previously sent
                # duplicated text to the API. The original line is already in
                # readline's history from when it was typed in the main REPL loop,
                # so pressing Up retrieves it for in-place editing with no custom hook.
                if isinstance(user_content, list):
                    original_text = next(
                        (b.get('text', '') for b in user_content
                         if isinstance(b, dict) and b.get('type') == 'text'),
                        ""
                    )
                    image_note = " (any attached images will be kept)"
                else:
                    original_text = user_content or ""
                    image_note = ""
                if original_text:
                    print(f"  Original message{image_note}: {original_text!r}")
                else:
                    print(f"  Original message had no text{image_note}.")
                print("  (Press the Up arrow to recall it from history if you want to edit it in place.)")
                new_text = input("[edit] You: ").strip()

                if isinstance(user_content, list):
                    kept = [b for b in user_content
                            if not (isinstance(b, dict) and b.get('type') == 'text')]
                    new_content = kept + ([{"type": "text", "text": new_text}] if new_text else [])
                    if not new_content:
                        print("Nothing to send; choose again.")
                        continue
                else:
                    if not new_text:
                        print("No text entered; choose again.")
                        continue
                    new_content = new_text
                return ('edit', new_content)

            elif choice == '2':
                return ('continue', None)
            elif choice == '3':
                return ('fallback', None)
            elif choice == '4':
                return ('cancel', None)
            else:
                print("Please enter 1, 2, 3, or 4.")

    def send_message(self, user_input):
        """Send a message and get response with streaming."""
        if not self.conversation:
            self.new_conversation()

        # Make the current model known to the tool executor (image resize cap)
        if self.tool_executor:
            self.tool_executor.model = self.conversation.model

        # Add user message, prepending any images queued via /image (image-first)
        if self.pending_images:
            user_content = list(self.pending_images)
            if user_input:
                user_content.append({"type": "text", "text": user_input})
            self.pending_images = []
        else:
            user_content = user_input
        self.conversation.messages.append({"role": "user", "content": user_content})
        self.conversation.unsaved_changes = True

        total_tool_calls = 0
        # Reset consecutive shell call counter for escalating sleep
        if self.tool_executor:
            self.tool_executor.consecutive_shell_calls = 0
        turn_input_tokens = 0
        turn_output_tokens = 0
        turn_cache_creation = 0
        turn_cache_read = 0

        rollback_point = len(self.conversation.messages) - 1  # Before user message we just added
        try:
            # Outer retry loop: normally runs once. Re-entered only after a
            # stop_reason == "refusal" response, if the user chooses to edit
            # their message or fall back to another model.
            while True:
                # Resolve per attempt: the model can change between attempts
                # (e.g. a fallback retry on claude-opus-4-8).
                thinking_kwargs, thinking_note = resolve_thinking_request(
                    self.conversation.model,
                    self.config.get('thinking_mode', 'auto'),
                    self.config.get('thinking_budget', 0),
                )
                if thinking_note:
                    print(f"\n{thinking_note}")
                print("\nClaude: ", end="", flush=True)

                refused = False
                while True:
                    # Prepare messages with cache control
                    prepared = prepare_messages_for_cache(
                        self.conversation.messages,
                        self.cache_ttl
                    )

                    # Build request
                    request_kwargs = {
                        "model": self.conversation.model,
                        "max_tokens": int(self.config.get('max_tokens', 8192)),
                        "messages": prepared,
                    }
                    if self.conversation.system_prompt:
                        request_kwargs["system"] = self.conversation.system_prompt

                    # Add tools if any are enabled
                    tools = self._build_tools_list()
                    if tools:
                        request_kwargs["tools"] = tools

                    # Add extended thinking if enabled (resolved once above, before the loop)
                    if thinking_kwargs:
                        request_kwargs["thinking"] = thinking_kwargs

                    # Make streaming API call
                    with self.client.messages.stream(**request_kwargs) as stream:
                        response_text = ""
                        thinking_text = ""
                        in_thinking = False

                        # Stream events to handle both thinking and text
                        for event in stream:
                            if hasattr(event, 'type'):
                                if event.type == 'content_block_start':
                                    block = getattr(event, 'content_block', None)
                                    if block and getattr(block, 'type', None) == 'thinking':
                                        in_thinking = True
                                        print("[Thinking...] ", end="", flush=True)
                                    elif block and getattr(block, 'type', None) == 'text':
                                        in_thinking = False
                                elif event.type == 'content_block_delta':
                                    delta = getattr(event, 'delta', None)
                                    if delta:
                                        if getattr(delta, 'type', None) == 'thinking_delta':
                                            thinking_text += getattr(delta, 'thinking', '')
                                        elif getattr(delta, 'type', None) == 'text_delta':
                                            text = getattr(delta, 'text', '')
                                            print(text, end="", flush=True)
                                            response_text += text
                                elif event.type == 'content_block_stop':
                                    if in_thinking:
                                        print(f"[{len(thinking_text)} chars]", flush=True)
                                        print("Claude: ", end="", flush=True)
                                        in_thinking = False

                        # Get the final message for metadata
                        response = stream.get_final_message()

                    self.stats['requests'] += 1

                    # Update stats
                    if hasattr(response, 'usage'):
                        self._update_stats(response.usage)
                        # Track per-turn totals for display
                        turn_input_tokens += response.usage.input_tokens
                        turn_output_tokens += response.usage.output_tokens
                        turn_cache_creation += getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                        turn_cache_read += getattr(response.usage, 'cache_read_input_tokens', 0) or 0

                    # A safety-classifier decline: HTTP 200 with stop_reason == "refusal",
                    # not an exception. Any partial content already streamed is incomplete
                    # per Anthropic's docs, so hand off to the recovery prompt instead of
                    # silently storing it.
                    if getattr(response, 'stop_reason', None) == 'refusal':
                        refused = True
                        break

                    # Check for tool use
                    tool_uses = []
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == 'tool_use':
                            tool_uses.append(block)

                    # If no tool use, we're done
                    if response.stop_reason != "tool_use" or not tool_uses:
                        # Save assistant response to history
                        # Always use _serialize_content to preserve thinking blocks
                        self.conversation.messages.append({
                            "role": "assistant",
                            "content": self._serialize_content(response.content)
                        })
                        break

                    # Handle tool use
                    # First, save assistant's response with tool_use blocks
                    self.conversation.messages.append({
                        "role": "assistant",
                        "content": self._serialize_content(response.content)
                    })

                    # Execute each tool and collect results
                    tool_results = []
                    for tool_use in tool_uses:
                        total_tool_calls += 1
                        self.stats['tool_calls'] += 1

                        # Show what tool is being called
                        print(f"\n[Tool: {tool_use.name}({tool_use.input})]", flush=True)

                        # Execute the tool
                        if self.tool_executor:
                            result = self.tool_executor.execute(tool_use.name, tool_use.input)
                        else:
                            result = f"Error: Tool executor not configured"


                        # Check for excessive tool call rate
                        # (removed: rate warning - escalating sleep handles this now)
                        # Show abbreviated result
                        if isinstance(result, list):
                            result_preview = "[image]"
                        else:
                            result_preview = result[:100] + "..." if len(result) > 100 else result
                        print(f"[Result: {result_preview}]", flush=True)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result
                        })

                    # Add tool results as a user message
                    self.conversation.messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    # Continue the loop - Claude will respond to tool results

                if not refused:
                    break  # normal completion - fall through to final stats

                # --- Handle the refusal: discard whatever this attempt appended
                # (including any tool round trips earlier in this same turn) and
                # let the user choose how to proceed. ---
                action, new_content = self._prompt_refusal_recovery(
                    response, user_content, response_text=response_text, thinking_text=thinking_text
                )

                if action == 'continue':
                    self.conversation.messages.append({
                        "role": "assistant",
                        "content": self._serialize_content(response.content)
                    })
                    print("[Keeping the partial response above as the final answer for this turn.]")
                    break

                if action == 'edit':
                    self.conversation.messages = self.conversation.messages[:rollback_point]
                    user_content = new_content
                    self.conversation.messages.append({"role": "user", "content": user_content})
                    continue

                if action == 'fallback':
                    self.conversation.messages = self.conversation.messages[:rollback_point]
                    self.conversation.messages.append({"role": "user", "content": user_content})
                    self.conversation.model = "claude-opus-4-8"
                    if self.tool_executor:
                        self.tool_executor.model = self.conversation.model
                    print("[Switching this conversation's model to claude-opus-4-8 and retrying.]")
                    continue

                # action == 'cancel'
                self.conversation.messages = self.conversation.messages[:rollback_point]
                print("Cancelled — message not sent.")
                return

            # Show final stats
            if hasattr(response, 'usage'):
                # Context size = input_tokens + cache_read (cache_read is part of context but billed differently)
                cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                context_size = response.usage.input_tokens + cache_read

                cache_info = ""
                if turn_cache_creation > 0:
                    cache_info = f", cache write: {turn_cache_creation:,}"
                elif turn_cache_read > 0:
                    cache_info = f", cache read: {turn_cache_read:,}"

                web_info = ""
                server_tool_use = getattr(response.usage, 'server_tool_use', None)
                if server_tool_use and getattr(server_tool_use, 'web_search_requests', 0):
                    web_info = f", web: {server_tool_use.web_search_requests}"

                tools_info = ""
                if total_tool_calls > 0:
                    tools_info = f", tools: {total_tool_calls}"

                # Show context size (how close to 200k limit) and output tokens
                print(f"\n[ctx: {context_size:,}, out: {turn_output_tokens:,}{cache_info}{web_info}{tools_info}, stop: {response.stop_reason}]")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            # Rollback all messages added during this failed send
            self.conversation.messages = self.conversation.messages[:rollback_point]

    def run(self):
        """Main REPL loop."""
        self.init_client()
        
        name = self.conversation.name if self.conversation else "(new)"
        model = self.conversation.model if self.conversation else self.config.get('default_model')
        web_status = "on" if self.web_search_enabled else "off"
        
        print(f"\ngraft - conversation: {name}")
        tools_status = "on" if self.tools_enabled else "off"
        print(f"model: {model}, cache: {self.cache_ttl}, web: {web_status}, tools: {tools_status}")
        print("Type /help for commands\n" + "-" * 40)
        
        while True:
            try:
                prompt = f"\n[{self.conversation.name or 'new'}] You: "
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                if self.conversation and self.conversation.unsaved_changes:
                    resp = input("Unsaved changes. Quit anyway? [y/N] ").strip().lower()
                    if resp != 'y':
                        continue
                break
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                if not self.handle_command(user_input):
                    break
            else:
                self.send_message(user_input)

# === Entry Point ===

def main():
    ensure_graft_dirs()
    save_default_config()
    env_vars = load_dotenv()
    
    config = load_config()
    setup_readline(config)
    
    session = GraftSession(config, env_vars)
    
    # Parse command line
    args = sys.argv[1:]
    
    if not args:
        # Interactive: show menu or start new
        convos = list_conversations()
        if convos:
            print("Saved conversations:")
            print(format_conversation_list(convos))
            print()
            choice = input("Load conversation (name), or press Enter for new: ").strip()
            if choice:
                session.load_conversation(choice)
            else:
                session.new_conversation()
        else:
            session.new_conversation()
    
    elif args[0] == '--list':
        convos = list_conversations()
        print(format_conversation_list(convos))
        return
    
    elif args[0] == '--import':
        if len(args) < 2:
            print("Usage: graft --import <file.json> [name] [--no-thinking] [--no-tool-use]")
            sys.exit(1)
        
        # Parse import arguments - defaults are to INCLUDE everything
        import_path = args[1]
        import_name = None
        include_thinking = '--no-thinking' not in args
        include_tool_use = '--no-tool-use' not in args
        
        # Find name argument (first non-flag after path)
        for arg in args[2:]:
            if not arg.startswith('--'):
                import_name = arg
                break
        
        if not session.import_conversation(import_path, import_name, include_thinking, include_tool_use):
            sys.exit(1)
    
    elif args[0].startswith('-'):
        print(f"Unknown option: {args[0]}")
        print("Usage: graft [name | --list | --import <file> [name]]")
        sys.exit(1)
    
    else:
        # Load by name
        if not session.load_conversation(args[0]):
            sys.exit(1)
    
    session.run()

if __name__ == '__main__':
    main()
