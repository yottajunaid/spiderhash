import sys
import ctypes
import platform
import threading
import time
import base64
from datetime import datetime

import utils 
import gui
import constants

def _print_log(message, log_type="INFO", is_error=False):
    timestamp = datetime.now().strftime('%H:%M:%S')
    prefix_parts = [f"[{timestamp}]"]
    if is_error:
        prefix_parts.append("[ERROR]")
    elif log_type and log_type.upper() != "INFO":
        prefix_parts.append(f"[{log_type.upper()}]")
    
    prefix = "".join(prefix_parts)
    print(f"{prefix} {message}")


def main():

    if platform.system() == "Windows":
        try:
            awareness = ctypes.c_int()
            errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            if errorCode == 0 and awareness.value != 2: 
                if ctypes.windll.shcore.SetProcessDpiAwareness(2) != 0: 
                    ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except AttributeError: 
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except AttributeError:
                _print_log("Failed to set DPI awareness (user32.SetProcessDPIAware not found).", is_error=True)
            except Exception as e_dpi_user32:
                 _print_log(f"Error setting DPI awareness via user32: {e_dpi_user32}", is_error=True)
        except Exception as e_dpi:
            _print_log(f"Error setting DPI awareness: {e_dpi}", is_error=True)

    stop_event = threading.Event()
    root_window = None
    try:
        root_window = gui.create_gui(stop_event) 
        
        if root_window:
            root_window.mainloop() 
        else:
            _print_log("GUI creation failed, root window not returned.", is_error=True)

    except KeyboardInterrupt:
        _print_log("Application interrupted by user (KeyboardInterrupt).")
        if not stop_event.is_set(): 
            stop_event.set()
    except Exception as e:
        _print_log(f"An unexpected error occurred in main: {e}", is_error=True)
        import traceback
        traceback.print_exc() 
        if not stop_event.is_set():
            stop_event.set()
    finally:
        _print_log("Application shutting down...")
        if not stop_event.is_set(): 
            stop_event.set()
        
        if hasattr(gui, 'root') and gui.root and hasattr(gui.root, 'registry') and 'rainbow_manager' in gui.root.registry:
            try:
                gui.root.registry['rainbow_manager'].cleanup()
                _print_log("Rainbow manager cleanup called.")
            except Exception as e_cleanup:
                _print_log(f"Error during rainbow manager cleanup: {e_cleanup}", is_error=True)
        
        _print_log("Shutdown complete.")

if __name__ == "__main__":
    main()