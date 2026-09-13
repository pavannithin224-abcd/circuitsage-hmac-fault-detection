# Security

Do not load untrusted `.joblib` or pickle files. Python object deserialization
can execute code. Verify the released model against `SHA256SUMS` before loading
it. Please report suspected credential exposure or malicious artifacts
privately to the repository maintainer instead of opening a public issue.
