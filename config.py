import os

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Log Parsing 
# Supported log levels treated as failure indicators
FAILURE_LEVELS  = {"ERROR", "CRITICAL", "TRACE", "WARNING"}
CRITICAL_LEVELS = {"ERROR", "CRITICAL"}

# OpenStack log regex pattern (raw log format)
OPENSTACK_LOG_PATTERN = (
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<component>[\w\.]+)\s+"
    r"(?P<content>.*)"
)

# Components considered critical for failure analysis
CRITICAL_COMPONENTS = {
    "nova.compute.manager",
    "nova.scheduler.manager",
    "nova.conductor.manager",
    "nova.network.manager",
    "nova.virt.libvirt.driver",
    "nova.compute.resource_tracker",
    "cinder.volume.manager",
    "neutron.agent.linux",
    "keystonemiddleware.auth_token",
}

# Transaction Building
TIME_WINDOW_SECONDS    = 65      # was 60 — 60s → 15 near-identical windows → lift≈1.0
                                # 5s  → ~177 windows of ~11 entries → real item variance
SESSION_WINDOW_SECONDS = 270    
MIN_EVENTS_PER_WINDOW  = 6        

# Apriori / FP-Growth
MIN_SUPPORT    = 0.11    # failure events occur in ~18% of windows; 0.15 captures them
MIN_CONFIDENCE = 0.5     
MIN_LIFT       = 2.65    # was 1.20 — dense log data rarely exceeds 1.2; 1.05 is meaningful
MAX_ITEMSET_LEN = 2      
MAX_FREQUENT_ITEMSETS = 50_000  # hard cap: skip rule generation above this count

# Failure Detection
SEVERITY_WEIGHTS = {
    "CRITICAL": 10,
    "ERROR":     7,
    "WARNING":   3,
    "INFO":      1,
    "DEBUG":     0,
}

# Patterns that always flag as high-severity failures
KNOWN_FAILURE_KEYWORDS = [
    "aborted", "failed", "failure", "exception", "error",
    "timeout", "refused", "unreachable", "lost", "down",
    "killed", "segfault", "panic", "overflow", "deadlock",
    "nova.exception", "no valid host", "build of instance",
    "lost connection", "service unavailable",
]

# Evaluation
TOP_N_PATTERNS   = 20   
ANOMALY_THRESHOLD = 2.5