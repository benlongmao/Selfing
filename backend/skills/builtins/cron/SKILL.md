---
name: cron
description: Scheduled reminders and recurring tasks
version: "2.2"
always_load: true
---

# Scheduled tasks skill

Scheduled tasks are durable reminders. Use ``schedule_task`` / ``schedule_list`` / ``schedule_cancel``.

**Frequency kinds:** ``once`` / ``hourly`` / ``daily`` / ``weekly`` / ``every_s`` / ``cron``

**Quick examples**
- Hourly ping:
  ``schedule_task(name="Hourly check", task_type="reminder", scheduled_time="now", description="...", frequency="hourly")``
- Fixed interval:
  ``frequency="every_s", every_seconds=300``
- Cron:
  ``frequency="cron", cron_expr="0 18 * * *"``

You decide when to act. Overdue items are surfaced in the prompt when the scheduler injects them.
