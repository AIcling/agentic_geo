"""
Configuration loader module for API keys and URLs.
Reads configuration from config.ini file in the project root using standard INI format.
"""
import os
import configparser

# Default values
DEFAULT_API_BASE = 'https://api.openai.com/v1'
DEFAULT_LOCAL_LLM_BASE = 'http://localhost:8000/v1'

# Cache for loaded config
_config_cache = None

def load_config(config_file='config.ini'):
    """
    Load configuration from an INI file.
    Uses Python's standard configparser module.
    
    Args:
        config_file: Path to the configuration file (relative to project root or absolute path)
    
    Returns:
        dict: Dictionary containing configuration values
    """
    global _config_cache
    
    # Use cached config if available
    if _config_cache is not None:
        return _config_cache
    
    config = {}
    
    # Try multiple possible locations for config file
    # 1. If absolute path provided, use it directly
    if os.path.isabs(config_file):
        config_path = config_file
    else:
        # 2. Try in current working directory
        config_path = os.path.join(os.getcwd(), config_file)
        if not os.path.exists(config_path):
            # 3. Try in project root (parent of src directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, config_file)
        if not os.path.exists(config_path):
            # 4. Try in src directory (where config_loader.py is)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_file)
    
    # If config file doesn't exist, try to use environment variables as fallback
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using environment variables.")
        config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
        config['OPENAI_API_BASE'] = os.environ.get('OPENAI_API_BASE', DEFAULT_API_BASE)
        config['USE_LOCAL_LLM'] = os.environ.get('USE_LOCAL_LLM', 'False').lower() == 'true'
        config['LOCAL_LLM_BASE'] = os.environ.get('LOCAL_LLM_BASE', DEFAULT_LOCAL_LLM_BASE)
        config['LOCAL_LLM_MODEL'] = os.environ.get('LOCAL_LLM_MODEL', 'local-model')
        config['USE_ASYNC'] = os.environ.get('USE_ASYNC', 'True').lower() == 'true'
        _config_cache = config
        return config
    
    # Read and parse INI config file
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding='utf-8')
    
    # Get values from [API] section
    if 'API' in parser:
        config['OPENAI_API_KEY'] = parser.get('API', 'OPENAI_API_KEY', fallback='')
        config['OPENAI_API_BASE'] = parser.get('API', 'OPENAI_API_BASE', fallback=DEFAULT_API_BASE)
        config['USE_LOCAL_LLM'] = parser.getboolean('API', 'USE_LOCAL_LLM', fallback=False)
        config['LOCAL_LLM_BASE'] = parser.get('API', 'LOCAL_LLM_BASE', fallback=DEFAULT_LOCAL_LLM_BASE)
        config['LOCAL_LLM_MODEL'] = parser.get('API', 'LOCAL_LLM_MODEL', fallback='local-model')
        config['USE_ASYNC'] = parser.getboolean('API', 'USE_ASYNC', fallback=True)
    
    # Environment variables override config file values (higher priority)
    # Only override if environment variable is explicitly set
    if 'LOCAL_LLM_MODEL' in os.environ:
        config['LOCAL_LLM_MODEL'] = os.environ['LOCAL_LLM_MODEL']
    
    # Set defaults if not specified
    if not config.get('OPENAI_API_KEY'):
        config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
    if not config.get('OPENAI_API_BASE'):
        config['OPENAI_API_BASE'] = os.environ.get('OPENAI_API_BASE', DEFAULT_API_BASE)
    if 'USE_LOCAL_LLM' not in config:
        config['USE_LOCAL_LLM'] = os.environ.get('USE_LOCAL_LLM', 'False').lower() == 'true'
    if 'LOCAL_LLM_BASE' not in config:
        config['LOCAL_LLM_BASE'] = os.environ.get('LOCAL_LLM_BASE', DEFAULT_LOCAL_LLM_BASE)
    if 'LOCAL_LLM_MODEL' not in config:
        config['LOCAL_LLM_MODEL'] = os.environ.get('LOCAL_LLM_MODEL', 'local-model')
    
    # Environment variables override config file values (higher priority)
    # This allows runtime override without modifying config.ini
    if 'LOCAL_LLM_MODEL' in os.environ:
        config['LOCAL_LLM_MODEL'] = os.environ['LOCAL_LLM_MODEL']
    if 'LOCAL_LLM_BASE' in os.environ:
        config['LOCAL_LLM_BASE'] = os.environ['LOCAL_LLM_BASE']
    if 'USE_LOCAL_LLM' in os.environ:
        config['USE_LOCAL_LLM'] = os.environ.get('USE_LOCAL_LLM', 'False').lower() == 'true'
    
    if 'USE_ASYNC' not in config:
        config['USE_ASYNC'] = os.environ.get('USE_ASYNC', 'True').lower() == 'true'
    
    _config_cache = config
    return config

def get_api_key():
    """Get OpenAI API key from config."""
    config = load_config()
    api_key = config.get('OPENAI_API_KEY', '')
    if not api_key and not config.get('USE_LOCAL_LLM', False):
        raise ValueError("OPENAI_API_KEY not found in config.txt or environment variables. Please set it in config.txt")
    return api_key

def get_api_base():
    """Get OpenAI API base URL from config."""
    config = load_config()
    return config.get('OPENAI_API_BASE', DEFAULT_API_BASE)

def use_local_llm():
    """Check if local LLM should be used."""
    config = load_config()
    return config.get('USE_LOCAL_LLM', False)

def get_local_llm_base():
    """Get local LLM base URL from config."""
    config = load_config()
    return config.get('LOCAL_LLM_BASE', DEFAULT_LOCAL_LLM_BASE)

def get_local_llm_model():
    """Get local LLM model name from config."""
    config = load_config()
    return config.get('LOCAL_LLM_MODEL', 'local-model')

def use_async():
    """Check if async mode should be used."""
    config = load_config()
    return config.get('USE_ASYNC', True)

def get_effective_api_config():
    """Get the effective API configuration based on USE_LOCAL_LLM setting."""
    config = load_config()
    if config.get('USE_LOCAL_LLM', False):
        # For local LLM, prefer OPENAI_API_BASE from environment (set by run_experiment.sh)
        # This allows the script to dynamically set the port
        base_url = os.environ.get('OPENAI_API_BASE') or config.get('LOCAL_LLM_BASE', DEFAULT_LOCAL_LLM_BASE)
        return {
            'api_key': 'dummy-key',  # Local LLM may not need real key
            'base_url': base_url,
            'default_model': config.get('LOCAL_LLM_MODEL', 'local-model'),
        }
    else:
        return {
            'api_key': get_api_key(),
            'base_url': config.get('OPENAI_API_BASE', DEFAULT_API_BASE),
            'default_model': None,  # Will use model specified in function calls
        }

def reload_config():
    """Force reload configuration from file and environment variables."""
    global _config_cache
    _config_cache = None
    return load_config()

