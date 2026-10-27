# Privacy

Nydra is open-source software. **Whoever runs an instance is responsible for the data on it.** This page describes what the software itself stores and does, so operators and users know what to expect. If you use someone else's hosted instance, ask the operator about their own practices too.

## What Nydra stores

| Data | Where | Notes |
|---|---|---|
| Account: username, email, password | Database | Passwords are stored hashed, never in plain text. With Google sign-in, the email, Google ID and profile image are stored as well. |
| Files you upload (tables, text, documents, images, archives) | Server disk (`UPLOAD_DIR`) | Analysis runs on the server. Files are not shared with other users. |
| Job records and their results | Database | One record per scan, including the analysis result. |
| Generated outputs (cleaned CSV, cleaned image zip, image report) | Server disk (`REPORT_DIR`) | Removed automatically after 7 days. |
| LLM provider keys you add in Settings | Database | Encrypted at rest. Only a masked version is ever shown back to you. |
| Nydra API keys you create | Database | Only a SHA-256 hash is stored. |
| Server logs | Server | Requests are logged with IP address and path. Tokens and passwords are not logged. |

## What leaves the server

- **Chat with an LLM provider:** if you configure a provider key and use the chat, your messages and an excerpt of your analysis results are sent to that provider. Check the provider's own privacy terms. Nothing is sent to a provider if you don't use the chat.
- Uploaded files themselves are not sent to any third party by Nydra.

## How long data is kept

| Data | Retention |
|---|---|
| Generated outputs and reports | 7 days |
| Temporary work folders | 24 hours |
| Unfinished chunked uploads | 24 hours |
| Uploaded files, job records, results | Until you delete them (see below) |
| Account | Until deleted on request (see below) |

## Deleting your data

- **Uploaded files:** `DELETE /api/v1/files/{file_id}` removes the file from the server disk (unless another of your uploads is the identical file).
- **A scan and its result:** `DELETE /api/v1/jobs/{job_id}/data` permanently removes the job record, its stored result and its generated files. A running job must be cancelled first.
- **API keys:** `DELETE /api/v1/auth/api-keys/{key_id}`.
- **Account deletion:** there is no self-service button yet. Open an issue in this repository's Issues tab asking for account deletion and give **only your username** (issues are public, so do not post your email or any personal data). The operator removes the account and its data.

The web interface does not have delete buttons yet; today deletion is available through the API (see the API docs of your instance).

## Not yet covered

- Backups: this version does not create backups. If an operator adds them, deleted data may remain in older backups until they expire.
- Files deleted before real deletion was added (the older "hide only" behaviour) may still be on disk; operators can remove them manually.

## Security

See [SECURITY.md](SECURITY.md) for protections and for how to report a vulnerability privately.

## Questions

Open an issue in this repository. For security problems, use private vulnerability reporting instead of a public issue.