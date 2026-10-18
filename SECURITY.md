# Security Policy

Nydra is an open-source data intelligence platform. It handles user uploads (tables, images, text, archives), so we take security seriously.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

- Preferred: use GitHub's private reporting on the repository
  (Security tab -> "Report a vulnerability").

Please include: what you found, steps to reproduce, the affected version or commit, and the impact you expect. You do not need a working exploit.

What to expect:

- Acknowledgement within **7 days**.
- A fix or a clear mitigation plan as soon as we can confirm the issue.
- Credit in the release notes if you want it.

This is a small project maintained by one person, so response times are best effort.

## Supported versions

Only the latest release on `main` receives security fixes.

## What is covered

Reports about any of the following are in scope:

- Authentication and sessions (JWT access/refresh tokens, Google sign-in, API keys)
- WebSocket authentication (token is sent in the first message, never in the URL)
- Authorization (one user reading or changing another user's files, jobs or results)
- File upload and archive handling (zip/tar extraction, path traversal, archive bombs, filename handling)
- Rate limiting and abuse protection (login lockout, per-IP and per-user limits)
- Resource limits (per-user storage quota, concurrent job caps, queue cap)
- Injection through uploaded content (for example CSV formula injection in cleaned files)

## Protections currently in place

| Area | Protection |
|---|---|
| Accounts | Case-insensitive usernames and emails; input length limits on register and login |
| Login | Per-IP limit plus failure-only lockout per account (8 failures / 5 min) |
| Other auth routes | Rate limits on refresh, Google exchange, API-key creation |
| WebSockets | First-message token auth, 5 s timeout, close code 4001 on failure |
| Uploads | Filename sanitizing on direct and chunked paths; stored names are server-generated |
| Archives | Safe zip/tar extraction (no `..`, absolute paths, symlinks; entry, size and ratio limits) |
| Quotas | Per-user storage quota, active-job cap per user, global queue cap |
| Heavy routes | Shared rate limit on image cleaning and report generation |
| Downloads | Reports are served as attachments; ownership checked on every download |
| Retention | Generated reports and temporary folders are deleted automatically (default 7 days) |

## Out of scope

- Denial of service by sheer traffic volume against a self-hosted instance
- Vulnerabilities in third-party dependencies with no demonstrated impact on Nydra (report those upstream)
- Social engineering, physical access, or attacks needing a compromised user device
- Findings that require a user to deliberately disable a protection in their own deployment

## Known limitations

We list these openly so you do not have to guess:

- Job state is held in memory, so a restart loses in-flight status (stale jobs are marked failed at startup).
- Behind a reverse proxy, per-IP limits need the proxy's forwarded-IP headers configured, otherwise all users share one bucket.
- An attacker who knows a username can trigger a 5-minute lockout of that account with repeated bad logins.
- Text analysis quality is English-only; this is a quality limit, not a security one.

## Deploying your own instance

- Set strong, unique secrets through environment variables; never commit `.env`.
- Serve over HTTPS only.
- Set storage and job limits (`MAX_USER_STORAGE_MB`, `MAX_ACTIVE_JOBS_PER_USER`, `MAX_QUEUE_DEPTH`, `MAX_FILE_MB`, `MAX_CHUNKED_GB`) to match your hardware.
- Keep dependencies updated.