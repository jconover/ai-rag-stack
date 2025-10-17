# n8n Workflow Automation for AI RAG Stack

This directory contains pre-built n8n workflows for automating documentation updates and maintenance tasks.

## Available Workflows

### 1. Weekly Documentation Update (`weekly-doc-update.json`)

**Recommended for most users** - Runs every Sunday at 2 AM.

**Features:**
- ✅ Updates all documentation repositories
- ✅ Automatically re-ingests if changes detected
- ✅ Sends Slack notification with update summary
- ✅ Sends email notification (optional)
- ✅ Logs all update events to JSON file
- ✅ Detailed reporting of which repos were updated

**Why Weekly?**
- Documentation doesn't change that frequently
- Reduces unnecessary processing
- Lower infrastructure overhead
- Gives you time to review changes

**Schedule:** `0 2 * * 0` (Every Sunday at 2 AM)

### 2. Daily Documentation Update (`daily-doc-update.json`)

**For high-frequency environments** - Runs every night at 2 AM.

**Features:**
- ✅ Updates all documentation repositories daily
- ✅ Automatically re-ingests if changes detected
- ✅ **Silent mode** - Only notifies if updates found
- ✅ Slack notification on updates only
- ✅ Lighter weight than weekly workflow

**When to Use:**
- You're working on bleeding-edge projects
- You need the latest docs daily
- You prefer proactive updates

**Schedule:** `0 2 * * *` (Every day at 2 AM)

---

## Setup Instructions

### Prerequisites

1. **n8n Running**: You need n8n installed and running
2. **AI RAG Stack**: Your RAG stack should be operational
3. **Notifications** (Optional): Slack webhook or email SMTP configured

### Step 1: Deploy n8n

Add n8n to your `docker-compose.yml`:

```yaml
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n-workflows:/workflows  # Mount workflows directory
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=changeme
      - TZ=America/New_York  # Set your timezone
    restart: unless-stopped

volumes:
  n8n_data:
```

Start n8n:

```bash
docker compose up -d n8n
```

Access n8n at: http://localhost:5678

### Step 2: Import Workflow

1. Open n8n at http://localhost:5678
2. Click **"Workflows"** in the left sidebar
3. Click **"Import from File"**
4. Select `weekly-doc-update.json` or `daily-doc-update.json`
5. Click **"Import"**

### Step 3: Configure Notifications

#### Option A: Slack Notifications

1. Create a Slack Incoming Webhook:
   - Go to https://api.slack.com/apps
   - Create a new app
   - Enable "Incoming Webhooks"
   - Add webhook to your channel
   - Copy the webhook URL

2. In the workflow, click the **"Send Slack Notification"** node
3. Replace `https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK` with your actual webhook URL

#### Option B: Email Notifications

1. In n8n, go to **Credentials** → **New Credential** → **SMTP**
2. Configure your SMTP settings:
   - Host: `smtp.gmail.com` (or your provider)
   - Port: `587`
   - Username: your email
   - Password: app-specific password
   - Secure: `true`

3. In the workflow, click the **"Send Email Notification"** node
4. Update email addresses:
   - From: `noreply@yourdomain.com`
   - To: `your-email@example.com`

#### Option C: No Notifications

If you don't want notifications:
1. Delete the "Send Slack Notification" node
2. Delete the "Send Email Notification" node
3. Connect "Format Success Message" directly to "Log Update Event"

### Step 4: Customize Paths

Update the path to your AI RAG Stack in the **"Update Documentation"** node:

```bash
cd /home/justin/Code/ai-rag-stack && make update-docs
```

Replace `/home/justin/Code/ai-rag-stack` with your actual path.

### Step 5: Activate Workflow

1. Click the **toggle switch** at the top right to activate
2. The workflow is now scheduled and will run automatically

---

## Testing the Workflow

### Manual Test Run

1. Open the workflow in n8n
2. Click **"Execute Workflow"** at the bottom
3. Watch the execution in real-time
4. Check for any errors

### Test Update Script Manually

Before relying on automation, test manually:

```bash
# Test the update script
make update-docs

# Expected output:
# - List of repositories checked
# - Notification if updates found
# - Auto re-ingestion if needed
```

---

## Workflow Details

### Weekly Workflow Flow

```
┌─────────────────────┐
│ Every Sunday 2 AM   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Update Docs Script  │
│ (make update-docs)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check Exit Code     │
│ 0 = Updates Found   │
│ 1 = No Updates      │
└──────┬──────┬───────┘
       │      │
   Yes │      │ No
       │      │
       ▼      ▼
   ┌─────┐ ┌─────────────┐
   │ Re- │ │ No Updates  │
   │ingest│ │ Message     │
   └──┬──┘ └──────┬──────┘
      │           │
      ▼           ▼
   ┌──────────────────┐
   │ Send Slack/Email │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ Log Event to File│
   └──────────────────┘
```

### Daily Workflow Flow (Simplified)

```
┌─────────────────────┐
│ Every Day 2 AM      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Update Docs         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Updates Found?      │
└──────┬──────────────┘
       │
   Yes │ (Only notify if yes)
       │
       ▼
┌─────────────────────┐
│ Send Slack (silent) │
└─────────────────────┘
```

---

## Notification Examples

### Slack Notification (Updates Found)

```
📚 Documentation Update Complete!

✓ 3 repositories updated:
  • Kubernetes
  • Docker
  • Rust

Re-ingestion completed successfully.
```

### Slack Notification (No Updates)

```
✓ Documentation Update Check Complete

No updates found. All documentation is current.

Next check: Next Sunday at 2 AM
```

### Email Subject

```
AI RAG Stack - Documentation Update Report
```

---

## Update Log

The workflow logs all update events to:

```
/home/justin/Code/ai-rag-stack/data/update-log.json
```

**Log Format:**
```json
[
  {
    "timestamp": "2025-10-17T02:00:00.000Z",
    "updatedCount": 3,
    "updatedRepos": ["Kubernetes", "Docker", "Rust"],
    "success": true
  },
  {
    "timestamp": "2025-10-24T02:00:00.000Z",
    "updatedCount": 0,
    "updatedRepos": [],
    "success": true
  }
]
```

**View logs:**
```bash
cat data/update-log.json | python3 -m json.tool
```

---

## Customization Options

### Change Schedule

Edit the cron expression in the **"Every Sunday at 2 AM"** node:

| Schedule | Cron Expression | Description |
|----------|----------------|-------------|
| Weekly (Sunday 2 AM) | `0 2 * * 0` | Recommended |
| Bi-weekly (1st & 15th) | `0 2 1,15 * *` | Mid-month updates |
| Monthly (1st of month) | `0 2 1 * *` | Light usage |
| Daily (2 AM) | `0 2 * * *` | High frequency |
| Weekdays only | `0 2 * * 1-5` | Business days |

### Add More Notifications

You can add additional notification channels:

- **Discord**: Use HTTP Request node with Discord webhook
- **Teams**: Use HTTP Request node with Teams webhook
- **Telegram**: Use Telegram node
- **SMS**: Use Twilio node
- **PagerDuty**: For critical updates

### Custom Logic

Add custom nodes to:
- Check vector database size before/after
- Compare performance metrics
- Run additional validation
- Archive old documentation versions
- Send summary reports

---

## Troubleshooting

### Workflow Not Running

**Check:**
1. Is the workflow **activated**? (toggle switch on)
2. Is n8n running? `docker ps | grep n8n`
3. Check n8n logs: `docker logs n8n`
4. Is timezone set correctly in docker-compose?

### Update Script Fails

**Check:**
1. Does the path exist? `ls /home/justin/Code/ai-rag-stack`
2. Is the update script executable? `chmod +x scripts/update_docs.sh`
3. Are git repositories valid? `cd data/docs/kubernetes && git status`
4. Run manually: `make update-docs`

### Notifications Not Sending

**Slack:**
- Is webhook URL correct?
- Test with curl:
  ```bash
  curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"Test"}' \
    YOUR_WEBHOOK_URL
  ```

**Email:**
- Are SMTP credentials valid?
- Check credential configuration in n8n
- Test with a simple email send workflow

### Re-ingestion Takes Too Long

If re-ingestion is slow (>30 minutes):
1. Consider running updates during off-peak hours
2. Increase backend resources in docker-compose
3. Use incremental ingestion (future enhancement)

---

## Advanced: Manual Trigger via API

You can trigger updates on-demand via n8n API:

```bash
# Get workflow ID from n8n UI
WORKFLOW_ID="your-workflow-id"

# Trigger manually
curl -X POST http://localhost:5678/api/v1/workflows/$WORKFLOW_ID/execute \
  -H "Content-Type: application/json" \
  -u "admin:changeme"
```

---

## Best Practices

### Weekly vs Daily

**Use Weekly if:**
- ✅ Documentation doesn't change frequently
- ✅ You want lower resource usage
- ✅ You prefer scheduled review windows

**Use Daily if:**
- ✅ Working on bleeding-edge projects
- ✅ Need latest security updates ASAP
- ✅ High-frequency development environment

### Notification Strategy

**Full Notifications (Weekly):**
- Always notifies (updates or not)
- Good for accountability
- Provides regular status updates

**Silent Mode (Daily):**
- Only notifies on updates
- Reduces notification fatigue
- Better for high-frequency checks

### Monitoring

Set up monitoring for:
- Failed workflow executions (n8n has built-in error workflows)
- Long-running ingestions (>1 hour)
- Storage usage growth
- Vector database size

---

## Security Considerations

### Credentials

- Never commit webhook URLs or SMTP passwords to git
- Use n8n's credential system
- Rotate credentials regularly

### Access Control

- Use strong n8n admin password
- Consider IP whitelisting for n8n
- Use HTTPS in production (reverse proxy)

### Script Permissions

- Update script runs with your user permissions
- Ensure proper file ownership of data/docs
- Consider using a service account

---

## Future Enhancements

Potential additions to these workflows:

- [ ] Incremental ingestion (only changed files)
- [ ] Pre/post update health checks
- [ ] Automatic rollback on ingestion failure
- [ ] Slack interactive buttons (approve/reject)
- [ ] Metrics dashboard integration
- [ ] Cost monitoring (storage growth)
- [ ] Performance comparison reports

---

## Support

If you encounter issues:

1. Check n8n execution logs in the UI
2. Run `make update-docs` manually to test
3. Review the troubleshooting section
4. Check n8n community forum: https://community.n8n.io

---

## Example Use Cases

### Use Case 1: Weekend Review

**Schedule:** Weekly (Sunday 2 AM)
**Notifications:** Slack + Email
**Workflow:** Review on Monday morning

### Use Case 2: Silent Monitoring

**Schedule:** Daily (2 AM)
**Notifications:** Slack (only on updates)
**Workflow:** Set and forget

### Use Case 3: Team Notification

**Schedule:** Weekly (Sunday 2 AM)
**Notifications:** Slack to #devops channel
**Workflow:** Team reviews updates together

---

**Status**: ✅ Ready to Deploy
**Version**: 1.0.0
**Last Updated**: 2025-10-17
