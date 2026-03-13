#!/usr/bin/env python3
"""
Tkinter GUI wrapper for conversational_bot.py

Changes made:
- process_queue now handles a "close" message and schedules window destruction on the main (GUI) thread.
- close_application accepts an optional delay (seconds) and enqueues a ("close", delay) message.
- run_bot schedules closing with a short delay after printing the final completion message,
  ensuring the final GUI output is visible before the window closes.
- on_timeout and other methods guard access to buttons that are commented out to avoid attribute errors.
"""
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import queue
import builtins
import time
import os
import importlib.util

class ConversationalBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Conversational Assistant")
        self.root.geometry("800x600")

        # Queue for thread-safe GUI updates
        self.msg_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.waiting_for_input = False
        self.current_prompt = ""
        self.timeout_id = None
        self.timeout_seconds = 120

        # Create GUI elements
        self.create_widgets()

        # Start message queue processor
        self.process_queue()

        # Load and start the bot
        self.load_bot_module()

    def create_widgets(self):
        """Create all GUI widgets"""
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            main_frame,
            text="RAG Conversational Assistant",
            font=("Arial", 16, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=(0, 10))

        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(output_frame, text="Conversation:", font=("Arial", 10, "bold")).pack(anchor="w")

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Consolas", 10),
            bg="#f8f9fa",
            fg="#2c3e50"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        self.output_text.config(state=tk.DISABLED)

        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.prompt_label = tk.Label(
            input_frame,
            text="Waiting...",
            font=("Arial", 9),
            fg="#7f8c8d"
        )
        self.prompt_label.pack(anchor="w")

        input_entry_frame = tk.Frame(input_frame)
        input_entry_frame.pack(fill=tk.X, pady=(5, 0))

        self.input_entry = tk.Entry(
            input_entry_frame,
            font=("Arial", 11),
            bg="white"
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.on_submit)

        self.submit_button = tk.Button(
            input_entry_frame,
            text="Submit",
            command=self.on_submit,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            state=tk.DISABLED
        )
        self.submit_button.pack(side=tk.LEFT)

        # (Optional buttons kept commented - handlers guard access)
        """
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        self.back_button = ...
        """

    def append_output(self, text, tag="normal"):
        """Append text to output area (thread-safe)"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def print_to_gui(self, text):
        """Thread-safe print to GUI"""
        self.msg_queue.put(("print", text))

    def request_input(self, prompt, timeout=None):
        """Request input from user (thread-safe). Blocks until input is available."""
        self.msg_queue.put(("input", prompt, timeout))
        return self.input_queue.get()

    def close_application(self, delay=0):
        """
        Request the main thread to close the application.
        delay: seconds to wait before destroying the window (scheduled on main thread).
        """
        try:
            # ensure delay is numeric
            d = float(delay) if delay is not None else 0.0
        except Exception:
            d = 0.0
        self.msg_queue.put(("close", d))

    def process_queue(self):
        """Process messages from the queue on the main (GUI) thread"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if not msg:
                    continue
                kind = msg[0]
                if kind == "print":
                    self.append_output(msg[1])
                elif kind == "input":
                    prompt = msg[1]
                    timeout = msg[2] if len(msg) > 2 else None
                    self.setup_input(prompt, timeout)
                elif kind == "close":
                    delay_seconds = msg[1] if len(msg) > 1 else 0
                    # schedule destruction on main thread after delay to let final prints show
                    self.root.after(int(max(0, float(delay_seconds)) * 1000), self._destroy_root)
                else:
                    # unknown message types can be ignored or logged
                    pass
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

    def _destroy_root(self):
        """Destroy root safely (main thread)."""
        try:
            if hasattr(self.root, "destroy"):
                self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

    def setup_input(self, prompt, timeout=None):
        """Setup input field for user input"""
        self.current_prompt = prompt
        self.waiting_for_input = True
        self.prompt_label.config(text=prompt)
        self.input_entry.config(state=tk.NORMAL)
        self.submit_button.config(state=tk.NORMAL)
        self.input_entry.focus()

        if timeout:
            try:
                # timeout is provided in seconds
                self.timeout_seconds = int(timeout)
            except Exception:
                # keep existing timeout_seconds if parsing fails
                pass
            self.start_timeout()

    def start_timeout(self):
        """Start input timeout"""
        if self.timeout_id:
            try:
                self.root.after_cancel(self.timeout_id)
            except Exception:
                pass
            self.timeout_id = None
        try:
            self.timeout_id = self.root.after(
                max(1, int(self.timeout_seconds)) * 1000,
                self.on_timeout
            )
        except Exception:
            self.timeout_id = None

    def on_timeout(self):
        """Handle input timeout"""
        if self.waiting_for_input:
            self.waiting_for_input = False
            try:
                self.input_entry.config(state=tk.DISABLED)
            except Exception:
                pass
            try:
                self.submit_button.config(state=tk.DISABLED)
            except Exception:
                pass

            # Guard commented-out/optional buttons (they may not exist)
            for btn_name in ("back_button", "exit_button", "leave_button"):
                btn = getattr(self, btn_name, None)
                if btn is not None:
                    try:
                        btn.config(state=tk.DISABLED)
                    except Exception:
                        pass

            self.input_queue.put(None)  # Signal timeout
            self.append_output(f"[TIMEOUT after {self.timeout_seconds} seconds]")

    def on_submit(self, event=None):
        """Handle submit button click"""
        if not self.waiting_for_input:
            return

        # Cancel timeout
        if self.timeout_id:
            try:
                self.root.after_cancel(self.timeout_id)
            except Exception:
                pass
            self.timeout_id = None

        # Get input
        user_input = ""
        try:
            user_input = self.input_entry.get()
        except Exception:
            user_input = ""
        try:
            self.input_entry.delete(0, tk.END)
        except Exception:
            pass

        # Disable input
        self.waiting_for_input = False
        try:
            self.input_entry.config(state=tk.DISABLED)
        except Exception:
            pass
        try:
            self.submit_button.config(state=tk.DISABLED)
        except Exception:
            pass

        # Show user input in output
        self.append_output(f"{self.current_prompt}{user_input}")

        # Send to bot thread
        self.input_queue.put(user_input)

    def quick_command(self, command):
        """Handle quick command buttons"""
        if self.waiting_for_input:
            if self.timeout_id:
                try:
                    self.root.after_cancel(self.timeout_id)
                except Exception:
                    pass
                self.timeout_id = None

            self.waiting_for_input = False
            try:
                self.input_entry.config(state=tk.DISABLED)
            except Exception:
                pass
            try:
                self.submit_button.config(state=tk.DISABLED)
            except Exception:
                pass

            self.append_output(f"{self.current_prompt}{command}")
            self.input_queue.put(command)

    def load_bot_module(self):
        """Load the conversational_bot module"""
        bot_path = os.path.join(os.path.dirname(__file__), "conversational_bot.py")

        if not os.path.exists(bot_path):
            messagebox.showerror(
                "Error",
                f"Could not find conversational_bot.py at {bot_path}\n"
                "Please ensure the file is in the same directory as this script."
            )
            self.append_output("ERROR: conversational_bot.py not found!")
            return

        bot_thread = threading.Thread(target=self.run_bot, args=(bot_path,), daemon=True)
        bot_thread.start()

    def run_bot(self, bot_path):
        """Run the bot in a separate thread and interact via queues"""
        original_print = None
        original_input = None
        original_pprint = None
        try:
            spec = importlib.util.spec_from_file_location("conversational_bot", bot_path)
            bot_module = importlib.util.module_from_spec(spec)

            # Redirect print/input to GUI-friendly versions
            original_print = builtins.print
            original_input = builtins.input

            def gui_print(*args, **kwargs):
                text = " ".join(str(arg) for arg in args)
                self.print_to_gui(text)

            def gui_input(prompt=""):
                return self.request_input(prompt)

            def gui_pprint(obj, *args, **kwargs):
                import pprint as pp_module
                formatted = pp_module.pformat(obj, *args, **kwargs)
                self.print_to_gui("=" * 60)
                self.print_to_gui("Available Default Queries:")
                self.print_to_gui("=" * 60)
                self.print_to_gui(formatted)
                self.print_to_gui("=" * 60)

            builtins.print = gui_print
            builtins.input = gui_input

            # Patch pprint if present
            try:
                import pprint as pprint_module
                original_pprint = getattr(pprint_module, "pprint", None)
                if original_pprint is not None:
                    pprint_module.pprint = gui_pprint
            except Exception:
                original_pprint = None

            # Load and execute the module
            spec.loader.exec_module(bot_module)

            # Run the app function if present
            if hasattr(bot_module, 'app'):
                try:
                    bot_module.app()
                except Exception as e:
                    self.print_to_gui(f"ERROR running bot app(): {e}")
            else:
                self.print_to_gui("ERROR: No 'app' function found in conversational_bot.py")

        except Exception as e:
            # Make sure exceptions are visible in the GUI
            import traceback
            self.print_to_gui(f"ERROR loading bot: {e}")
            self.print_to_gui(traceback.format_exc())
        finally:
            # Restore original functions
            try:
                if original_print is not None:
                    builtins.print = original_print
            except Exception:
                pass
            try:
                if original_input is not None:
                    builtins.input = original_input
            except Exception:
                pass

            if original_pprint is not None:
                try:
                    import pprint as pprint_module
                    pprint_module.pprint = original_pprint
                except Exception:
                    pass

            # Final GUI message and schedule app closure after a short delay so the user can see it
            self.print_to_gui("\n=== Bot execution completed ===")
            # schedule a closure after 2 seconds (gives GUI time to show the final message)
            self.close_application(delay=2.0)


def main():
    root = tk.Tk()
    app = ConversationalBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
