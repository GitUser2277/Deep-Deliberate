# DeepDeliberate Framework Examples

## 🚨 Development Status Notice

**⚠️ EXAMPLES FOR PLANNED FUNCTIONALITY**: These examples demonstrate the intended usage of the DeepDeliberate Framework. Currently, the framework is 65% complete with critical implementation gaps preventing the example commands from working.

**Current State**: Configuration files and agent examples are ready, but the core framework functionality requires completion.

**See**: [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) for detailed component status.

This directory contains example configurations, agents, and documentation to help you get started with the DeepDeliberate framework quickly.

## Directory Structure

```
examples/
├── agents/                     # Sample PydanticAI agents for different domains
│   ├── customer_service_agent.py
│   ├── education_agent.py
│   └── healthcare_agent.py
├── configs/                    # Example configuration files with domain-specific personas
│   ├── customer_service_config.json
│   ├── education_config.json
│   └── healthcare_config.json
├── docs/                       # Documentation and guides
│   ├── quick_start_tutorial.md
│   └── troubleshooting_guide.md
└── README.md                   # This file
```

## Quick Start

1. **Choose a domain** that matches your use case
2. **Copy the relevant files** to your working directory
3. **Set up your API key** in environment variables
4. **Run your first test**

### Example Commands

```bash
# Customer Service Testing
cp examples/agents/customer_service_agent.py my_agent.py
cp examples/configs/customer_service_config.json my_config.json
python deepdeliberate.py -file my_agent.py -config my_config.json -auto -count 5

# Education Testing
cp examples/agents/education_agent.py my_agent.py
cp examples/configs/education_config.json my_config.json
python deepdeliberate.py -file my_agent.py -config my_config.json -auto -count 5

# Healthcare Testing
cp examples/agents/healthcare_agent.py my_agent.py
cp examples/configs/healthcare_config.json my_config.json
python deepdeliberate.py -file my_agent.py -config my_config.json -auto -count 5
```

## Example Agents

### Customer Service Agent (`agents/customer_service_agent.py`)

A sample agent designed for customer support scenarios with:
- Professional, empathetic responses
- Context-aware personalization
- Support for different customer types
- Error handling and escalation patterns

**Key Features:**
- Customer context integration
- Account type awareness
- Professional tone maintenance
- Support history consideration

### Education Agent (`agents/education_agent.py`)

A tutoring agent that adapts to different learning levels:
- Grade-appropriate explanations
- Socratic method implementation
- Encouraging and supportive responses
- Subject-specific knowledge areas

**Key Features:**
- Student context integration
- Grade level adaptation
- Learning style consideration
- Educational best practices

### Healthcare Agent (`agents/healthcare_agent.py`)

A healthcare information agent with safety-first design:
- Evidence-based information
- Clear medical disclaimers
- Patient privacy protection
- Appropriate referral guidance

**Key Features:**
- Patient context awareness
- Safety-first responses
- Medical disclaimer inclusion
- Professional referral guidance

## Example Configurations

### Customer Service Configuration

**Personas included:**
- `satisfied_customer`: Generally happy customers with minor questions
- `frustrated_customer`: Customers experiencing issues or problems
- `technical_customer`: Technically proficient customers with advanced needs

**Template Parameters:**
- Features, services, products, error codes, time periods, issues

### Education Configuration

**Personas included:**
- `elementary_student`: Young learners (grades 1-5) with basic questions
- `middle_school_student`: Middle schoolers (grades 6-8) with homework help
- `high_school_student`: High schoolers (grades 9-12) preparing for college

**Template Parameters:**
- Science concepts, math topics, literature works, historical events, research topics

### Healthcare Configuration

**Personas included:**
- `general_health_inquirer`: People seeking general health information
- `concerned_patient`: Worried individuals with specific symptoms
- `chronic_condition_manager`: People managing ongoing health conditions

**Template Parameters:**
- Symptoms, conditions, medications, treatments, wellness areas

## Customization Guide

### Modifying Personas

To create personas for your specific use case:

1. **Identify your user types**: Who will interact with your agent?
2. **Define behavioral patterns**: How do they typically communicate?
3. **Set tone specifications**: What's their communication style?
4. **Create scenario templates**: What questions do they ask?
5. **Define evaluation criteria**: What matters most for your domain?

### Example Persona Customization

```json
{
  "name": "enterprise_customer",
  "description": "Large business customer with complex needs",
  "behavioral_patterns": [
    "Asks about enterprise features and scalability",
    "Concerned with security and compliance",
    "May involve multiple stakeholders in decisions",
    "Expects detailed technical documentation"
  ],
  "tone_specifications": {
    "formality": "professional",
    "patience": "moderate",
    "technical_level": "advanced",
    "emotion": "business-focused"
  },
  "scenario_templates": [
    "Inquire about {enterprise_feature} for {company_size} organization",
    "Ask about compliance with {regulation} requirements",
    "Request technical specifications for {integration_type}"
  ],
  "evaluation_criteria": {
    "accuracy": 0.4,
    "relevance": 0.3,
    "completion": 0.2,
    "safety": 0.1
  },
  "template_parameters": {
    "enterprise_feature": ["SSO integration", "bulk user management", "advanced analytics", "custom branding"],
    "company_size": ["500+ employees", "multinational", "distributed teams", "multiple departments"],
    "regulation": ["GDPR", "HIPAA", "SOX", "PCI DSS"],
    "integration_type": ["API integration", "webhook setup", "data export", "third-party connectors"]
  }
}
```

### Modifying Agents

To adapt agents for your domain:

1. **Update system prompts**: Reflect your domain expertise
2. **Add context models**: Include relevant user/session data
3. **Implement domain logic**: Add specific business rules
4. **Handle edge cases**: Plan for unusual scenarios

### Example Agent Customization

```python
class LegalContext(BaseModel):
    """Context for legal consultation scenarios."""
    client_id: str
    case_type: str
    jurisdiction: str
    urgency_level: str

legal_agent: Agent[LegalContext, str] = Agent(
    model,
    deps_type=LegalContext,
    system_prompt="""
    You are a legal information assistant providing general legal guidance.
    
    IMPORTANT: You provide information only, not legal advice.
    Always recommend consulting with qualified attorneys for specific legal matters.
    
    Guidelines:
    - Provide accurate legal information
    - Explain legal concepts clearly
    - Reference relevant laws and regulations
    - Maintain client confidentiality
    - Recommend appropriate legal resources
    """
)
```

## Testing Strategies

### Progressive Testing

1. **Start small**: Test with 3-5 iterations to verify setup
2. **Expand gradually**: Increase to 10-20 iterations for initial validation
3. **Full testing**: Run 50+ iterations for comprehensive evaluation

### Persona-Specific Testing

```bash
# Test each persona individually
python deepdeliberate.py -file agent.py -config config.json -persona satisfied_customer -auto -count 10
python deepdeliberate.py -file agent.py -config config.json -persona frustrated_customer -auto -count 10
python deepdeliberate.py -file agent.py -config config.json -persona technical_customer -auto -count 10
```

### Comparative Testing

```bash
# Test different agent versions
python deepdeliberate.py -file agent_v1.py -config config.json -auto -count 20 -output results/v1
python deepdeliberate.py -file agent_v2.py -config config.json -auto -count 20 -output results/v2

# Compare results
python -c "
import pandas as pd
v1 = pd.read_csv('results/v1/session_*.csv')
v2 = pd.read_csv('results/v2/session_*.csv')
print(f'V1 average score: {v1[\"evaluation_score\"].mean():.2f}')
print(f'V2 average score: {v2[\"evaluation_score\"].mean():.2f}')
"
```

## Best Practices

### Configuration Management

- **Version control**: Keep configurations in version control
- **Environment-specific configs**: Use different configs for dev/staging/prod
- **Validation**: Always validate configurations before running tests
- **Documentation**: Document persona rationale and expected behaviors

### Agent Development

- **Incremental testing**: Test agents frequently during development
- **Error handling**: Implement robust error handling
- **Logging**: Add detailed logging for debugging
- **Performance**: Monitor response times and resource usage

### Result Analysis

- **Trend monitoring**: Track scores over time
- **Persona comparison**: Compare performance across different user types
- **Failure analysis**: Investigate low-scoring interactions
- **Continuous improvement**: Use results to refine agents and personas

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/agent-testing.yml
name: Agent Testing
on: [push, pull_request]

jobs:
  test-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Test customer service agent
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          python deepdeliberate.py -file examples/agents/customer_service_agent.py \
            -config examples/configs/customer_service_config.json -auto -count 10
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results/
```

### Monitoring Script

```python
#!/usr/bin/env python3
"""
Daily agent performance monitoring script.
"""

import subprocess
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def run_agent_tests():
    """Run comprehensive agent tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"monitoring/results_{timestamp}"
    
    # Run tests for each domain
    domains = ['customer_service', 'education', 'healthcare']
    results = {}
    
    for domain in domains:
        cmd = [
            'python', 'deepdeliberate.py',
            '-file', f'examples/agents/{domain}_agent.py',
            '-config', f'examples/configs/{domain}_config.json',
            '-auto', '-count', '20',
            '-output', f'{output_dir}/{domain}'
        ]
        
        subprocess.run(cmd, check=True)
        
        # Analyze results
        df = pd.read_csv(f'{output_dir}/{domain}/session_latest.csv')
        results[domain] = {
            'avg_score': df['evaluation_score'].mean(),
            'min_score': df['evaluation_score'].min(),
            'total_tests': len(df)
        }
    
    return results

def send_alert_if_needed(results):
    """Send alert if scores drop below threshold."""
    threshold = 0.7
    alerts = []
    
    for domain, metrics in results.items():
        if metrics['avg_score'] < threshold:
            alerts.append(f"{domain}: {metrics['avg_score']:.2f} (below {threshold})")
    
    if alerts:
        # Send email alert (configure SMTP settings)
        msg = MIMEText(f"Agent performance alerts:\n" + "\n".join(alerts))
        msg['Subject'] = 'Agent Performance Alert'
        msg['From'] = 'monitoring@yourcompany.com'
        msg['To'] = 'team@yourcompany.com'
        
        # Configure and send email
        # smtp_server.send_message(msg)
        print("Alerts:", alerts)

if __name__ == "__main__":
    results = run_agent_tests()
    send_alert_if_needed(results)
    print("Monitoring complete:", results)
```

## Support and Resources

- **Documentation**: See the main project documentation for detailed API reference
- **Troubleshooting**: Check `docs/troubleshooting_guide.md` for common issues
- **Community**: Join our community for tips and best practices
- **Examples**: This directory contains working examples for immediate use

## Contributing

To contribute new examples:

1. **Follow existing patterns**: Use similar structure and documentation
2. **Test thoroughly**: Ensure examples work out of the box
3. **Document clearly**: Include clear explanations and use cases
4. **Consider domains**: Add examples for new domains or use cases

Happy testing! 🎯