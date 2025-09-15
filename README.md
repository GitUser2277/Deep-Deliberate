# DeepDeliberate Framework

**A lightweight, drag-and-drop AI agent testing framework for systematic evaluation of Pydantic-based AI agents through natural language interactions.**

## 🚨 Development Status

**⚠️ FRAMEWORK UNDER DEVELOPMENT**: This framework is currently 65% complete with excellent architecture but critical implementation gaps. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed progress.

**Current State**: Configuration system, CLI interface, error handling, and DeepSeek output enhancement components are production-ready. Core logic (API integration, query generation, evaluation) requires implementation to be functional.

**Recent Updates**: Enhanced DeepSeek R1 response parsing, reasoning content extraction, and CLI formatting systems have been implemented. See [AI_AGENT_HUB/claude_code_output.md](AI_AGENT_HUB/claude_code_output.md) for detailed enhancement status.

## 🎯 Core Value Proposition

**"One-time configuration setup + simple CLI execution = comprehensive AI agent testing with full automation or step-by-step control for any domain."**

## ✨ Key Features

- **🔌 Universal Compatibility**: Works with any PydanticAI agent with minimal integration
- **🎭 Persona-Based Testing**: Custom personas that simulate realistic user behavior patterns
- **⚡ Dual Execution Modes**: 
  - **Auto mode**: Fully automated batch testing
  - **Approve mode**: Step-by-step manual oversight with Continue/Retry/Exit options
- **🧠 AI-Powered Evaluation**: DeepSeek R1-powered evaluation with chain-of-thought reasoning
- **📊 Complete Audit Trail**: CSV logging with session recovery and detailed metadata
- **🌍 Cross-Platform**: Works on Windows, macOS, and Linux
- **🔒 Security-First**: Input sanitization, secure API handling, and comprehensive error management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DeepSeek API key
- PydanticAI agent to test

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deepdeliberate
   ```

2. **Set up Python environment**:
   ```bash
   # Using UV (recommended)
   uv venv
   uv pip install -r requirements.txt
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Verify setup** (framework not yet functional for full testing):
   ```bash
   # Test configuration system
   python -c "from deepdeliberate.config.manager import ConfigManager; print('✅ Config system ready')"
   
   # Test CLI interface
   python deepdeliberate.py --help
   
   # See IMPLEMENTATION_STATUS.md for current limitations
   ```

## 📖 Table of Contents

- [Installation Guide](#installation-guide)
- [Usage Examples](#usage-examples)
- [Configuration Guide](#configuration-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 📦 Installation Guide

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: Minimum 2GB RAM available
- **Storage**: 500MB free space for framework and logs

### Platform-Specific Installation

#### Windows

```powershell
# Install UV (recommended package manager)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and setup
git clone <repository-url>
cd deepdeliberate
uv venv
uv pip install -r requirements.txt

# Activate environment
.venv\Scripts\activate
```

#### macOS

```bash
# Install UV using Homebrew
brew install uv

# Clone and setup
git clone <repository-url>
cd deepdeliberate
uv venv
uv pip install -r requirements.txt

# Activate environment
source .venv/bin/activate
```

#### Linux (Ubuntu/Debian)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd deepdeliberate
uv venv
uv pip install -r requirements.txt

# Activate environment
source .venv/bin/activate
```

### Environment Configuration

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your settings**:
   ```bash
   # Required API Configuration
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_ENDPOINT=https://your-deepseek-endpoint.com
   
   # Optional: OpenAI/Azure OpenAI (for fallback)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_AZURE_ENDPOINT=https://your-azure-endpoint.com
   
   # Security Settings (recommended defaults)
   ENABLE_AGENT_VALIDATION=true
   SANDBOX_AGENT_EXECUTION=true
   ENABLE_DATA_REDACTION=true
   ```

3. **Verify installation**:
   ```bash
   # Test available components
   python -c "from deepdeliberate.config.manager import ConfigManager; print('✅ Configuration system ready')"
   python deepdeliberate.py --help
   
   # Note: Full framework functionality requires completion of implementation gaps
   # See IMPLEMENTATION_STATUS.md for details
   ```

---

## 🎮 Usage Examples

### ⚠️ Current Limitations

**These examples represent the intended functionality but are not yet operational due to implementation gaps:**

- Query generation requires DeepSeek API integration
- Agent execution needs completion of execution wrapper
- Response evaluation requires LLM integration
- Session recovery needs implementation

**See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for current working components.**

### Basic Usage (Planned Functionality)

#### Auto Mode (Fully Automated)
```bash
# Run 10 automated tests with default persona
python deepdeliberate.py -file src/my_agent.py -mode auto -count 10

# Run 25 tests with specific persona
python deepdeliberate.py -file src/my_agent.py -mode auto -count 25 -persona angry_customer

# Custom output directory
python deepdeliberate.py -file src/my_agent.py -mode auto -count 15 -output results/customer_service/
```

#### Approve Mode (Step-by-Step Control)
```bash
# Interactive testing with manual approval
python deepdeliberate.py -file src/my_agent.py -mode approve -count 10

# Test specific persona with approval
python deepdeliberate.py -file src/my_agent.py -mode approve -persona confused_user -count 5
```

### Advanced Usage

#### Custom Configuration
```bash
# Use custom config file
python deepdeliberate.py -file src/my_agent.py -mode auto -config custom_config.json

# Combine multiple options
python deepdeliberate.py \
  -file src/customer_service_agent.py \
  -mode approve \
  -persona enterprise_customer \
  -count 20 \
  -output enterprise_tests/ \
  -config enterprise_config.json
```

#### Session Recovery
```bash
# Framework automatically detects interrupted sessions
python deepdeliberate.py -file src/my_agent.py -mode auto -count 50

# If interrupted, restart with same command - you'll be prompted:
# "Found 1 interrupted session(s). Would you like to recover? [y/N]"
```

### Command Reference

| Parameter | Short | Required | Description | Example |
|-----------|-------|----------|-------------|---------|
| `--file` | `-file` | ✅ | Path to Python file with PydanticAI agents | `src/agent.py` |
| `--mode` | `-mode` | ✅ | Execution mode: `auto` or `approve` | `auto` |
| `--count` | `-count` | ❌ | Number of test queries (default: 10) | `25` |
| `--persona` | `-persona` | ❌ | Specific persona name from config | `angry_customer` |
| `--output` | `-output` | ❌ | Output directory (default: test_results) | `results/` |
| `--config` | `-config` | ❌ | Config file path (default: config.json) | `custom.json` |

---

## ⚙️ Configuration Guide

### Configuration File Structure

The `config.json` file controls all framework behavior. Here's the complete structure:

```json
{
  "personas": [...],
  "api_settings": {...},
  "security_settings": {...},
  "logging_settings": {...},
  "performance_settings": {...}
}
```

### Persona Configuration

Personas define how the framework generates test scenarios. Each persona simulates a specific user type:

```json
{
  "name": "angry_customer",
  "behavioral_patterns": [
    "Expresses frustration clearly",
    "Demands immediate solutions",
    "Uses emphatic language",
    "May escalate if not satisfied"
  ],
  "tone_specifications": {
    "formality": "informal",
    "patience": "low",
    "technical_level": "basic",
    "emotion": "frustrated"
  },
  "scenario_templates": [
    "Complain about {issue} not working properly",
    "Demand immediate fix for {problem}",
    "Express anger about {service_failure}"
  ],
  "evaluation_criteria": {
    "accuracy": 0.2,
    "relevance": 0.2,
    "completion": 0.3,
    "safety": 0.3
  }
}
```

#### Persona Fields Explained

- **`name`**: Unique identifier for the persona
- **`behavioral_patterns`**: List of behavioral characteristics
- **`tone_specifications`**: Key-value pairs defining communication style
- **`scenario_templates`**: Template strings for query generation (use `{variable}` for dynamic content)
- **`evaluation_criteria`**: Weights for evaluation dimensions (must sum to 1.0)

#### Creating Custom Personas

1. **Domain-Specific Personas**:
   ```json
   {
     "name": "medical_patient",
     "behavioral_patterns": [
       "Describes symptoms clearly",
       "Asks about treatment options",
       "Concerned about side effects"
     ],
     "tone_specifications": {
       "formality": "formal",
       "patience": "high",
       "technical_level": "low",
       "emotion": "concerned"
     },
     "scenario_templates": [
       "Ask about symptoms of {condition}",
       "Inquire about treatment for {ailment}",
       "Request information about {medication} side effects"
     ],
     "evaluation_criteria": {
       "accuracy": 0.4,
       "relevance": 0.3,
       "completion": 0.2,
       "safety": 0.1
     }
   }
   ```

2. **Technical User Personas**:
   ```json
   {
     "name": "developer_user",
     "behavioral_patterns": [
       "Asks detailed technical questions",
       "Expects code examples",
       "Comfortable with technical jargon"
     ],
     "tone_specifications": {
       "formality": "casual",
       "patience": "moderate",
       "technical_level": "high"
     },
     "scenario_templates": [
       "How do I implement {feature} in {language}?",
       "What's the best practice for {technical_concept}?",
       "Debug this {error_type} error"
     ],
     "evaluation_criteria": {
       "accuracy": 0.4,
       "relevance": 0.3,
       "completion": 0.2,
       "safety": 0.1
     }
   }
   ```

### API Settings

Configure external API connections:

```json
{
  "api_settings": {
    "deepseek_endpoint": "${DEEPSEEK_ENDPOINT}",
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "rate_limit_per_minute": 60,
    "verify_ssl": true,
    "enable_request_signing": true,
    "max_input_length": 1000
  }
}
```

### Security Settings

Control framework security behavior:

```json
{
  "security_settings": {
    "enable_agent_validation": true,
    "allow_dangerous_functions": false,
    "sandbox_agent_execution": true,
    "enable_data_redaction": true,
    "strict_mode": true
  }
}
```

### Logging Settings

Configure output and logging:

```json
{
  "logging_settings": {
    "output_directory": "test_results",
    "log_level": "INFO",
    "include_metadata": true,
    "encrypt_logs": false,
    "enable_audit_logging": true,
    "retention_days": 30
  }
}
```

### Performance Settings

Optimize framework performance:

```json
{
  "performance_settings": {
    "max_concurrent_requests": 5,
    "memory_limit_gb": 2,
    "cache_enabled": true
  }
}
```

---

## 🔧 Agent Integration

### PydanticAI Agent Requirements

Your agent must be a PydanticAI Agent instance. The framework automatically discovers agents in your Python file.

#### Basic Agent Example

```python
from pydantic_ai import Agent

# Simple agent
agent = Agent(
    'openai:gpt-4',
    system_prompt='You are a helpful customer service assistant.'
)

@agent.tool
def get_user_info(user_id: str) -> str:
    """Get user information by ID."""
    return f"User {user_id} information"
```

#### Advanced Agent Example

```python
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Optional

class CustomerServiceDeps(BaseModel):
    user_database: dict
    ticket_system: object

class ServiceResponse(BaseModel):
    message: str
    ticket_id: Optional[str] = None
    escalated: bool = False

# Agent with dependencies and structured output
customer_service_agent = Agent(
    'openai:gpt-4',
    deps_type=CustomerServiceDeps,
    result_type=ServiceResponse,
    system_prompt='''
    You are a customer service agent. Help users with their issues
    and create tickets when necessary.
    '''
)

@customer_service_agent.tool
def create_ticket(issue: str, priority: str) -> str:
    """Create a support ticket."""
    return f"Ticket created: {issue} (Priority: {priority})"
```

### Agent Discovery

The framework automatically finds PydanticAI agents in your file:

1. **Scans Python file** for `Agent` instances
2. **Extracts metadata**: model, output_type, deps_type
3. **Validates compatibility** with framework
4. **Initializes execution wrapper** for testing

### Supported Agent Features

- ✅ **Basic agents** with simple string responses
- ✅ **Structured output** with Pydantic models
- ✅ **Tool/function calling** agents
- ✅ **Dependency injection** via `deps_type`
- ✅ **Async and sync** execution modes
- ✅ **Custom system prompts**
- ✅ **Multiple agents** in single file

---

## 📚 API Reference

### Core Components

#### FrameworkCore

Main orchestration class that coordinates all framework components.

```python
class FrameworkCore:
    async def initialize(self, config: FrameworkConfig) -> None:
        """Initialize framework with configuration."""
        
    async def run_testing_session(self, cli_config: CLIConfig) -> SessionResult:
        """Execute a complete testing session."""
        
    async def recover_session(self, session_id: str) -> SessionState:
        """Recover an interrupted session."""
        
    async def shutdown(self) -> None:
        """Clean shutdown of framework components."""
```

#### ConfigManager

Handles configuration loading and validation.

```python
class ConfigManager:
    def load_config(self, path: str) -> FrameworkConfig:
        """Load and validate configuration from file."""
        
    def validate_personas(self, personas: List[Persona]) -> ValidationResult:
        """Validate persona configurations."""
        
    def get_persona(self, name: str) -> Persona:
        """Get specific persona by name."""
```

#### QueryGenerator

Generates test queries using AI.

```python
class DeepSeekQueryGenerator:
    async def generate_query(self, persona: Persona, context: Dict) -> GeneratedQuery:
        """Generate a single test query."""
        
    async def generate_edge_cases(self, persona: Persona) -> List[GeneratedQuery]:
        """Generate edge case scenarios."""
        
    async def retry_generation(self, failed_query: GeneratedQuery) -> GeneratedQuery:
        """Retry failed query generation."""
```

#### Evaluator

Evaluates agent responses.

```python
class DeepSeekEvaluator:
    async def evaluate_response(
        self, 
        query: str, 
        response: str, 
        persona: Persona
    ) -> EvaluationResult:
        """Evaluate agent response quality."""
        
    async def generate_reasoning(self, evaluation: EvaluationResult) -> str:
        """Generate explanation for evaluation score."""
```

#### SessionLogger

Handles logging and session persistence.

```python
class SessionLogger:
    async def log_interaction(self, interaction: TestInteraction) -> None:
        """Log a single test interaction."""
        
    async def create_checkpoint(self, session_state: SessionState) -> str:
        """Create session checkpoint for recovery."""
        
    async def recover_from_checkpoint(self, checkpoint_id: str) -> SessionState:
        """Recover session from checkpoint."""
```

### Data Models

#### CLIConfig

Configuration from command-line arguments.

```python
@dataclass
class CLIConfig:
    agent_file: str
    mode: ExecutionMode
    count: int
    persona: Optional[str] = None
    output_dir: Optional[str] = None
    config_file: str = "config.json"
```

#### TestInteraction

Single test interaction record.

```python
@dataclass
class TestInteraction:
    timestamp: datetime
    session_id: str
    persona_name: str
    query: str
    agent_response: str
    evaluation_score: float
    evaluation_reasoning: str
    execution_time: float
    metadata: Dict[str, Any]
```

#### EvaluationResult

Result of response evaluation.

```python
@dataclass
class EvaluationResult:
    overall_score: float
    dimension_scores: Dict[str, float]  # accuracy, relevance, completion, safety
    reasoning: str
    confidence: float
    evaluation_time: float
```

#### SessionResult

Complete session results.

```python
@dataclass
class SessionResult:
    session_id: str
    mode: ExecutionMode
    persona: Persona
    completed_interactions: List[TestInteraction]
    total_execution_time: float
    average_score: float
    success_rate: float
```

---

## 🔍 Output and Results

### CSV Log Format

Each test session generates a CSV file with detailed results:

```csv
timestamp,session_id,persona_name,query,agent_response,evaluation_score,evaluation_reasoning,execution_time,metadata
2024-01-15T10:30:00Z,sess_abc123,angry_customer,"I can't login to my account!","I understand your frustration...",8.5,"Response shows empathy and provides solution",2.3,"{""model"":""gpt-4""}"
```

#### CSV Columns Explained

- **`timestamp`**: ISO 8601 timestamp of interaction
- **`session_id`**: Unique session identifier
- **`persona_name`**: Persona used for query generation
- **`query`**: Generated test query
- **`agent_response`**: Agent's response to query
- **`evaluation_score`**: Overall score (0-10)
- **`evaluation_reasoning`**: AI explanation of score
- **`execution_time`**: Time taken for interaction (seconds)
- **`metadata`**: Additional context (JSON format)

### Session Recovery Files

Checkpoint files enable session recovery:

```json
{
  "session_id": "sess_abc123",
  "current_iteration": 15,
  "total_iterations": 50,
  "mode": "auto",
  "persona": {...},
  "completed_interactions": [...],
  "checkpoint_data": {...}
}
```

### Results Analysis

Use the CSV output for analysis:

```python
import pandas as pd

# Load results
df = pd.read_csv('test_results/sessions_20240115_103000.csv')

# Calculate statistics
avg_score = df['evaluation_score'].mean()
success_rate = (df['evaluation_score'] >= 7.0).mean()

# Analyze by persona
persona_stats = df.groupby('persona_name')['evaluation_score'].agg(['mean', 'std', 'count'])

# Find problematic queries
low_scores = df[df['evaluation_score'] < 5.0]
```

---

## 🚨 Troubleshooting

### Current Implementation Issues

#### 1. Framework Initialization Failures

**Problem**: CLI commands fail with component initialization errors

**Root Cause**: Critical implementation gaps in core components

**Status**: See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed breakdown

**Workaround**: Test individual components:
```bash
# Test configuration system
python -c "from deepdeliberate.config.manager import ConfigManager; print('Config OK')"

# Test CLI parsing
python deepdeliberate.py --help
```

#### 2. API Integration Not Functional

**Problem**: DeepSeek API calls return placeholder responses

**Root Cause**: API client methods contain `pass` statements instead of actual implementation

**Files Affected**: 
- `deepdeliberate/core/deepseek_client.py`
- `deepdeliberate/core/async_deepseek_client.py`

#### 3. Query Generation Non-Functional

**Problem**: Query generator returns `None` instead of generated queries

**Root Cause**: Core generation logic not implemented

**Files Affected**: `deepdeliberate/core/query_generator.py`

### Legacy Issues (Will Apply Once Framework is Functional)

#### 1. Agent Discovery Failures

**Problem**: "No PydanticAI agents found in file"

**Solutions**:
```python
# ✅ Correct: Agent instance at module level
from pydantic_ai import Agent
agent = Agent('openai:gpt-4', system_prompt='...')

# ❌ Incorrect: Agent inside function/class
def create_agent():
    return Agent('openai:gpt-4', system_prompt='...')
```

#### 2. API Authentication Errors

**Problem**: "DeepSeek API authentication failed"

**Solutions**:
1. Check `.env` file exists and has correct API key
2. Verify API key is valid and active
3. Check endpoint URL is correct
4. Ensure environment variables are loaded:
   ```bash
   # Test environment loading
   python -c "import os; print(os.getenv('DEEPSEEK_API_KEY'))"
   ```

#### 3. Configuration Validation Errors

**Problem**: "Invalid persona configuration"

**Solutions**:
1. Validate JSON syntax:
   ```bash
   python -c "import json; json.load(open('config.json'))"
   ```
2. Check evaluation criteria sum to 1.0:
   ```json
   "evaluation_criteria": {
     "accuracy": 0.25,
     "relevance": 0.25,
     "completion": 0.25,
     "safety": 0.25
   }
   ```
3. Ensure required fields are present

#### 4. Session Recovery Issues

**Problem**: "Cannot recover interrupted session"

**Solutions**:
1. Check checkpoint files exist in output directory
2. Verify file permissions for output directory
3. Clear corrupted checkpoints:
   ```bash
   rm test_results/checkpoint_*.json
   ```

#### 5. Memory Issues

**Problem**: "Out of memory during batch testing"

**Solutions**:
1. Reduce concurrent requests in config:
   ```json
   "performance_settings": {
     "max_concurrent_requests": 2
   }
   ```
2. Lower batch size:
   ```bash
   python deepdeliberate.py -file agent.py -mode auto -count 10
   ```
3. Enable memory monitoring:
   ```json
   "performance_settings": {
     "memory_limit_gb": 1
   }
   ```

### Platform-Specific Issues

#### Windows

**PowerShell Execution Policy**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path Issues**:
```powershell
# Use forward slashes or raw strings
python deepdeliberate.py -file "src/agent.py" -mode auto
```

#### macOS

**SSL Certificate Issues**:
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Permission Issues**:
```bash
chmod +x deepdeliberate.py
```

#### Linux

**Missing Dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# CentOS/RHEL
sudo yum install python3-devel python3-pip
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Run with verbose output
python deepdeliberate.py -file agent.py -mode auto -count 5
```

### Getting Help

1. **Check logs**: Look in `test_results/` for error logs
2. **Validate setup**: Run `python verify_setup.py`
3. **Test configuration**: Use minimal config first
4. **Check dependencies**: Ensure all packages installed correctly

---

## 🔮 Future Features

### Grounded Query Generation with MCP Integration

DeepDeliberate is exploring integration with Model Context Protocol (MCP) servers to enhance query generation with real-world context. This future feature would enable:

- **Real-world data integration**: Access approved social media sources, support forums, and community discussions
- **Grounded scenario generation**: Create test queries based on actual customer complaints and user experiences
- **Enhanced realism**: Generate more authentic test scenarios using real user language and pain points
- **Competitive intelligence**: Learn from industry-wide customer feedback patterns

See [future_features/grounded_query_gen_mcp.md](future_features/grounded_query_gen_mcp.md) for detailed specifications.

## 🤝 Contributing

### Development Setup

1. **Fork and clone**:
   ```bash
   git clone https://github.com/yourusername/deepdeliberate.git
   cd deepdeliberate
   ```

2. **Set up development environment**:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   uv pip install -e .
   ```

3. **Install development tools**:
   ```bash
   uv pip install pytest black mypy bandit safety
   ```

### Code Quality

Run quality checks before submitting:

```bash
# Format code
black deepdeliberate/

# Type checking
mypy deepdeliberate/

# Security scan
bandit -r deepdeliberate/

# Dependency security
safety check

# Run tests
pytest tests/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=deepdeliberate tests/
```

### Submitting Changes

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run quality checks
4. Commit with clear message
5. Push and create pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PydanticAI**: For the excellent agent framework
- **DeepSeek**: For powerful AI evaluation capabilities
- **Click**: For elegant CLI interface
- **Pydantic**: For robust data validation

---

## 📞 Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/deepdeliberate/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/deepdeliberate/discussions)

---

*DeepDeliberate Framework - Making AI agent testing systematic, reliable, and effortless.*

### Rate Limiting Diagnostics

The framework now provides enhanced rate limiting diagnostics to help you identify whether rate limits are being hit locally (client-side) or by Azure servers:

#### Message Types

**🛡️ LOCAL RATE LIMIT** - Your client is preventing API overload (GOOD)
```
🛡️ LOCAL RATE LIMIT: Blocking request to prevent API overload. Waiting 4.2s (Local limit: 60/min)
```
This means your local rate limiter is working correctly to prevent overwhelming Azure's API.

**🚫 AZURE API RATE LIMIT** - Azure rejected your request (BAD)  
```
🚫 AZURE API RATE LIMIT: Request rejected by DeepSeek/Azure servers! Server returned HTTP 429. 
Your local limit (60/min) is too aggressive for Azure's actual limits. Consider reducing API_RATE_LIMIT_PER_MINUTE in .env
```
This means Azure's servers are rejecting your requests - you need to reduce your rate limits.

#### Rate Limiting Configuration

Control rate limiting through your `.env` file:

```bash
# Rate limiting settings - adjust based on your Azure API limits
API_RATE_LIMIT_PER_MINUTE=15          # Start conservative (1 request every 4 seconds)
MAX_CONCURRENT_REQUESTS=2             # Limit simultaneous requests  
REQUEST_TIMEOUT_SECONDS=60            # Generous timeout for complex requests
```

#### Diagnostics Script

Run the enhanced diagnostics script to test your rate limiting:

```bash
python debug_config.py
```

This will:
- Test your current rate limiting settings
- Show you exactly where bottlenecks occur  
- Provide recommendations for optimal settings
- Display real-time rate limit statistics

#### Troubleshooting Rate Limits

1. **If you see 🛡️ LOCAL RATE LIMIT messages:**
   - This is GOOD - your client is preventing overload
   - No action needed, the system is working correctly

2. **If you see 🚫 AZURE API RATE LIMIT messages:**
   - This is BAD - Azure is rejecting your requests
   - Reduce `API_RATE_LIMIT_PER_MINUTE` in your `.env` file
   - Start with 15 and adjust downward if needed

3. **Finding the optimal rate limit:**
   - Start conservative (API_RATE_LIMIT_PER_MINUTE=10)
   - Gradually increase until you see Azure rejections
   - Set your limit slightly below the rejection point