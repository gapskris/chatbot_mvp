<img width="680" height="158" alt="image" src="https://github.com/user-attachments/assets/619fe546-d140-4698-8640-43e4e279e89e" />

<img width="710" height="303" alt="image" src="https://github.com/user-attachments/assets/96d96b3c-5f2a-4959-a905-0618f8849113" />

Architecture overview
	• Gateway/API: FastAPI with Auth token, rate limiting stub, payload normalization.
	• NLP: Deterministic intent detection and date entity extraction.
	• Knowledge: In-memory RAG index with cosine similarity.
	• Tools: LMS-style endpoints (enroll_course, fetch_progress).
	• Dialog policy: Deterministic next-action selection with confidence thresholds.
	• Memory: Per-session store (ephemeral), plus audit logs.
UI: Minimal HTML+JS chat widget that talks to the API.<img width="716" height="183" alt="image" src="https://github.com/user-attachments/assets/c0fceef7-7ccb-41d4-b7cc-59124142fee2" />


