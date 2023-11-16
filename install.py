import launch

if not launch.is_installed("transformers"):
    try:
        launch.run_pip("install transformers", "requirements for transformers")
    except Exception:
        print("Can't install transformers. Please follow the readme to install manually")
