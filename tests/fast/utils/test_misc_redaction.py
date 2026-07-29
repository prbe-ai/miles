from miles.utils.misc import _redact_command_for_logging


def test_redact_command_for_logging_hides_runtime_and_cli_secrets(monkeypatch):
    monkeypatch.setenv("PROBE_TOKEN", "probe-live-secret")
    command = (
        """ray job submit --runtime-env-json='{"env_vars":{"""
        """"PROBE_TOKEN":"probe-live-secret","MODAL_TOKEN_SECRET":"modal-secret"}}' """
        "--wandb-key 'wandb-secret' --agent-server-auth-token bridge-secret"
    )

    redacted = _redact_command_for_logging(command)

    for secret in ("probe-live-secret", "modal-secret", "wandb-secret", "bridge-secret"):
        assert secret not in redacted
    assert "<redacted:PROBE_TOKEN>" in redacted
    assert "<redacted:MODAL_TOKEN_SECRET>" in redacted
