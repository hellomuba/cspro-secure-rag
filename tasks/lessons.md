# Lessons

Patterns captured after a correction, so the same mistake is not repeated.

## 1. Check for work the user is already doing before starting a large download

**What happened (2026-11-19).** I started `docker compose up -d --build`, which
began pulling the ~3.2 GB `ollama/ollama` image, while the maintainer was
already pulling the same image in their own terminal. Two pulls competed for
bandwidth on a metered corporate connection and slowed down the thing that
actually mattered.

**Rule.** Before kicking off any multi-GB download, long install, or anything
that consumes a shared resource: check whether it is already running
(`docker ps`, `docker images`, running processes), and say what I am about to
download and roughly how big it is *before* starting it, not after.

**Corollary.** Prefer reusing what the user already has. The standalone
`aegisrag-ollama` container on `127.0.0.1:11434` serves the native workflow
directly, with no compose stack needed at all.

## 2. Container names are a shared namespace

My `compose.yaml` used `container_name: aegisrag-ollama`, the same name the
maintainer's standalone `docker run` used. That would have made compose fail to
start or fight over the model volume. Fixed by prefixing compose-managed
containers (`aegisrag-compose-*`).

**Rule.** Pinning `container_name` trades Docker's automatic namespacing for a
collision risk. Either leave it unset, or prefix it distinctly enough that it
cannot clash with a hand-run container.

## 3. Priorities shift; re-read them rather than finishing the current thread

When the maintainer flagged an imminent deadline and said to stop Docker work,
the right move was to stop immediately, make the minimum change needed to leave
Docker in a non-broken state, and return to the assessment-critical path --
not to finish the Docker verification I had already started.

## 4. Never let a normalisation choice silently defeat an abstention threshold

Found during implementation, not by correction, but worth keeping. BM25 scores
normalised by *matched* idf mass let a question made entirely of unknown words
("what is the canary value for tenant B") score ~1.0 on the strength of one
near-universal term, which would have defeated the retrieval threshold and sent
an unauthorised-looking query to the model. Charging absent query terms the
maximum idf fixed it.

**Rule.** When a security control depends on a numeric threshold, write the
adversarial input as a test *first* and confirm the number actually moves.

## 5. Report what ran, not what was set up to run

Several deliverables here (live inference, Docker, LoRA) can be *prepared*
without being *executed*. The README status table uses explicit
VERIFIED / NOT VERIFIED / UNEXECUTED markers filled in only after a real run,
and integration tests skip rather than pass when a runtime is absent, so a
green suite can never be mistaken for evidence a model ran.
