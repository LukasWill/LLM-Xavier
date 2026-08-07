# Security and privacy

LLM-X accepts credentials only through an environment-variable name such as
`OPENAI_API_KEY`. It does not serialize credential values, configured endpoints,
absolute input/output paths, usernames or hostnames into result files.

Raw model responses are deliberately not written to the public result schema. Parsed
predictions, parse status, metrics and—when enabled—complete prompts are stored. Prompts
contain trajectory data and should therefore only be published when the underlying
trajectory is publishable.

The public evaluator does not load policy checkpoints. Do not load untrusted pickle-based
PyTorch checkpoints in auxiliary research code.

Report security or privacy problems through a private security advisory on the eventual
public repository. Do not include credentials or private datasets in a public issue.
