"""Aider CLI provider implementation.

Adds aider (https://aider.chat) as a CAO provider. Like every other provider
this is just "launch the CLI in the tmux pane, then read state back out of the
captured buffer" — aider happens to be a prompt_toolkit REPL that prints a
banner, model/repo info, then an empty ``>`` prompt when it is idle.

Credentials: the launch command points aider at an ``--env-file`` (written
out-of-band, chmod 600) carrying OPENAI_API_BASE / OPENAI_API_KEY so the API
key never appears in the tmux command line or the captured session log. The
model comes from the agent profile (e.g. ``openai/as/gpt-5.5``); litellm's
``openai/`` prefix routes the request at the OpenAI-compatible gateway base.
"""

import asyncio
import logging
import re
import shlex
from typing import Optional

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.constants import CAO_HOME_DIR
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status
from cli_agent_orchestrator.utils.text import strip_terminal_escapes

logger = logging.getLogger(__name__)

# Env file carrying OPENAI_API_BASE / OPENAI_API_KEY for the gateway. Written
# out-of-band (chmod 600) before launch so the key never lands in the tmux
# command line or the captured scrollback.
AIDER_ENV_FILE = str(CAO_HOME_DIR / "aider.env")

DEFAULT_AIDER_MODEL = "openai/as/gpt-5.5"

# Empty idle prompt. aider labels the prompt by edit-format/mode — ``>``,
# ``architect>``, ``code>``, ``ask>`` — so allow an optional leading word.
IDLE_PROMPT_PATTERN = r"^\s*(?:\w+\s*)?>\s*$"
# A submitted user message echoed at the prompt: ``> do the thing``.
USER_PROMPT_PATTERN = r"^\s*(?:\w+\s*)?>\s+\S"
# Markers aider prints after a real turn (token/cost summary, edit results).
TURN_MARKER_PATTERN = (
    r"(?:Tokens:\s|Applied edit|Added .+ to the chat|Wrote |Created |"
    r"Edited |Committing|No changes made)"
)
# Confirmation prompts. We launch with --yes-always so these should auto-
# resolve; detect them defensively so a stray one surfaces as WAITING.
WAITING_PROMPT_PATTERN = r"(?:\(Y\)es/\(N\)o|\(y/n\)|\[Yes\]:|Allow .*\?|Run shell command)"
ERROR_PATTERN = r"(?:^Error:|^ERROR:|Traceback \(most recent call last\):|litellm\.\w*Error|APIError)"

# Lines to inspect at the tail of the captured buffer for prompt/spinner state.
TAIL_LINES = 10


class AiderProvider(BaseProvider):
    """Provider for the aider CLI."""

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[list] = None,
        skill_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(terminal_id, session_name, window_name, allowed_tools, skill_prompt)
        self._initialized = False
        self._agent_profile = agent_profile
        self._model = model or DEFAULT_AIDER_MODEL

    @property
    def paste_enter_count(self) -> int:
        # prompt_toolkit accepts the (possibly multi-line) bracketed paste on a
        # single Enter.
        return 1

    @property
    def paste_submit_delay(self) -> float:
        return 0.5

    def _build_aider_command(self) -> str:
        """Build the aider launch command.

        ``--no-pretty``/``--no-stream`` make responses land as a single clean
        block (no live markdown re-render) so COMPLETED detection is reliable;
        ``--yes-always`` keeps the non-interactive flow from blocking on
        confirmations; ``--map-tokens 0`` skips the repo-map on this tiny tree.
        """
        parts = [
            "aider",
            "--model",
            self._model,
            "--weak-model",
            self._model,
            "--edit-format",
            "diff",
            "--env-file",
            AIDER_ENV_FILE,
            "--no-pretty",
            "--no-stream",
            "--no-auto-commits",
            "--no-dirty-commits",
            "--no-gitignore",
            "--no-check-update",
            "--no-show-release-notes",
            "--no-show-model-warnings",
            "--no-suggest-shell-commands",
            "--no-detect-urls",
            "--map-tokens",
            "0",
            "--yes-always",
        ]
        return shlex.join(parts)

    async def initialize(self) -> bool:
        from cli_agent_orchestrator.services.status_monitor import status_monitor

        if not await wait_for_shell(self.terminal_id, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        # Warm up the shell before launching aider — fresh tmux shells can drop
        # the first interactive program otherwise (same as the codex provider).
        status_monitor.notify_input_sent(self.terminal_id)
        get_backend().send_keys(self.session_name, self.window_name, "echo ready")
        await asyncio.sleep(2.0)

        command = self._build_aider_command()
        status_monitor.notify_input_sent(self.terminal_id)
        get_backend().send_keys(self.session_name, self.window_name, command)

        # aider loads litellm + the model on first launch; give it room.
        if not await wait_until_status(
            self.terminal_id,
            {TerminalStatus.IDLE, TerminalStatus.COMPLETED},
            timeout=120.0,
            polling_interval=1.0,
        ):
            raise TimeoutError("Aider initialization timed out after 120 seconds")

        self._initialized = True
        return True

    def get_status(self, output: str) -> TerminalStatus:
        if not output:
            return TerminalStatus.UNKNOWN

        clean = strip_terminal_escapes(output)
        lines = clean.splitlines()

        last_nonblank = ""
        for ln in reversed(lines):
            if ln.strip():
                last_nonblank = ln
                break
        tail = "\n".join(lines[-TAIL_LINES:])

        # An empty ``>`` prompt at the very bottom means aider is ready.
        if re.match(IDLE_PROMPT_PATTERN, last_nonblank):
            had_turn = bool(
                re.search(USER_PROMPT_PATTERN, clean, re.MULTILINE)
                or re.search(TURN_MARKER_PATTERN, clean)
            )
            return TerminalStatus.COMPLETED if had_turn else TerminalStatus.IDLE

        # Not at the idle prompt: a blocking confirmation, an error, or aider is
        # still generating.
        if re.search(WAITING_PROMPT_PATTERN, tail):
            return TerminalStatus.WAITING_USER_ANSWER
        if re.search(ERROR_PATTERN, tail, re.MULTILINE):
            return TerminalStatus.ERROR
        return TerminalStatus.PROCESSING

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Return aider's reply to the most recent user message.

        The reply is the text between the last ``> <message>`` line and the
        following empty ``>`` prompt.
        """
        clean = strip_terminal_escapes(script_output)
        user_matches = list(re.finditer(USER_PROMPT_PATTERN, clean, re.MULTILINE))
        if user_matches:
            last_user = user_matches[-1]
            line_end = clean.find("\n", last_user.start())
            body_start = line_end + 1 if line_end != -1 else last_user.end()
            rest = clean[body_start:]
            idle_after = re.search(IDLE_PROMPT_PATTERN, rest, re.MULTILINE)
            end = body_start + idle_after.start() if idle_after else len(clean)
            text = clean[body_start:end].strip()
            if text:
                return text
        raise ValueError("No aider response found")

    def exit_cli(self) -> str:
        return "/exit"

    def cleanup(self) -> None:
        self._initialized = False
