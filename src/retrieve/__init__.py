import os

# Create directories and files
directories = [
    "cite",
    "eval",
    "app"
]

files = {
    "cite": [
        "__init__.py",
        "format_apa.py",
        "format_ieee.py",
        "bibtex.py"
    ],
    "eval": [
        "__init__.py",
        "build_ground_truth.py",
        "metrics.py",
        "run_eval.py"
    ],
    "app": [
        "__init__.py",
        "cli.py",
        "api.py",
        "streamlit_app.py"
    ]
}

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    for file in files[directory]:
        with open(os.path.join(directory, file), 'w') as f:
            f.write("# This is the " + file + " file.\n")
