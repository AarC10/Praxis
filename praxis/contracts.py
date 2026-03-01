from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SkillContract:
    name: str
    language: str
    runtime: str
    required_functions: List[str]
    function_signatures: Dict[str, Dict]
    allowed_imports: List[str]
    forbidden_imports: List[str]
    forbidden_calls: List[str]
    max_lines: int = 500
    timeout_seconds: int = 30

STANDALONE_PYTHON = SkillContract(
    name="standalone_python",
    language="python",
    runtime="standalone",
    required_functions=["run"],
    function_signatures={
        "run": {
            "params": ["context"],
            "returns": "dict"
        }
    },
    allowed_imports=[
        "numpy", "cv2", "PIL", "matplotlib",
        "scipy", "sklearn", "math", "json",
        "time", "datetime", "collections"
    ],
    forbidden_imports=[
        "os", "sys", "subprocess", "importlib",
        "eval", "exec", "compile", "__import__"
    ],
    forbidden_calls=[
        "eval", "exec", "compile", "open",
        "input", "raw_input"
    ]
)

ROS_PYTHON = SkillContract(
    name="ros_python",
    language="python",
    runtime="ros2",
    required_functions=["run"],
    function_signatures={
        "run": {
            "params": ["context"],
            "returns": "dict"
        }
    },
    # ROS SKills will need os/sys for fpaths and rclpy for messaging
    # Shouldnt expect direct rclpy calls and data should be already serialzied in ctx.params
    allowed_imports=[
        "numpy", "cv2", "PIL", "matplotlib",
        "scipy", "sklearn", "math", "json",
        "time", "datetime", "collections",
        "os", "sys", "pathlib",
        "rclpy",
        "sensor_msgs", "geometry_msgs", "std_msgs",
        "nav_msgs", "cv_bridge", "tf2_ros",
        "transforms3d", "quaternion",
    ],
    forbidden_imports=[
        "subprocess", "importlib",
        "eval", "exec", "compile", "__import__",
    ],
    forbidden_calls=[
        "eval", "exec", "compile",
        "input", "raw_input",
    ],
    max_lines=1000,
    timeout_seconds=60,
)

CONTRACTS = {
    "standalone_python": STANDALONE_PYTHON,
    "ros_python": ROS_PYTHON,
}