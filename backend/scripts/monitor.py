#!/usr/bin/env python3
"""
Monitoring and management script for automated MiniGPT training
Handles session recovery, progress tracking, and automatic restarts
"""

import os
import json
import time
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys


class AutoTrainingMonitor:
    def __init__(self):
        self.is_running = False
        self.current_process = None
        self.start_time = None
        self.session_duration_hours = 8
        self.cooldown_minutes = 30
        self.max_restarts = 5
        self.restart_count = 0

        # Create monitoring directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("monitoring", exist_ok=True)

        self.log_file = "logs/monitor.log"
        self.status_file = "monitoring/status.json"

    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().isoformat()
        log_msg = f"[{timestamp}] MONITOR: {message}"
        print(log_msg)

        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def save_status(self, status):
        """Save current status to file"""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "restart_count": self.restart_count,
            "session_duration_hours": self.session_duration_hours
        }

        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    def load_status(self):
        """Load previous status"""
        try:
            with open(self.status_file, "r") as f:
                data = json.load(f)

            self.restart_count = data.get("restart_count", 0)
            self.log(f"Loaded previous status - restart count: {self.restart_count}")
            return data

        except FileNotFoundError:
            self.log("No previous status found")
            return {}

    def start_training_session(self):
        """Start an automated training session"""
        self.log("Starting new automated training session")
        self.is_running = True
        self.start_time = datetime.now()
        self.save_status("training_started")

        try:
            # Start the auto training script
            cmd = [
                "python", "scripts/auto_train.py",
                "--config", "configs/small.yaml",
                "--iterations", "50",
                "--recover"  # Always try to recover previous state
            ]

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            self.log(f"Training process started with PID: {self.current_process.pid}")

            # Monitor the process
            self.monitor_training_process()

        except Exception as e:
            self.log(f"Failed to start training: {str(e)}")
            self.is_running = False
            self.save_status("failed_to_start")

    def monitor_training_process(self):
        """Monitor the training process and handle session limits"""
        session_end_time = self.start_time + timedelta(hours=self.session_duration_hours)

        while self.is_running and self.current_process:
            current_time = datetime.now()

            # Check if process is still alive
            if self.current_process.poll() is not None:
                # Process ended
                return_code = self.current_process.returncode
                self.log(f"Training process ended with code: {return_code}")

                if return_code == 0:
                    self.log("Training completed successfully")
                    self.save_status("training_completed")
                else:
                    self.log("Training failed")
                    self.save_status("training_failed")

                self.is_running = False
                break

            # Check session time limit
            if current_time >= session_end_time:
                self.log("Session time limit reached, gracefully stopping")
                self.stop_training_gracefully()
                break

            # Log progress periodically
            time_remaining = session_end_time - current_time
            if time_remaining.seconds % 1800 == 0:  # Every 30 minutes
                self.log(f"Training in progress. Time remaining: {time_remaining}")
                self.save_status("training_in_progress")

            # Check memory usage and system health
            self.check_system_health()

            time.sleep(60)  # Check every minute

    def check_system_health(self):
        """Check system resources and training health"""
        try:
            # Check if log file is being updated (sign of active training)
            log_path = Path("logs/auto_trainer.log")
            if log_path.exists():
                mod_time = datetime.fromtimestamp(log_path.stat().st_mtime)
                time_since_update = datetime.now() - mod_time

                if time_since_update > timedelta(minutes=30):
                    self.log("Warning: Training log hasn't been updated in 30 minutes")
                    # Could implement recovery logic here

        except Exception as e:
            self.log(f"Error checking system health: {str(e)}")

    def stop_training_gracefully(self):
        """Gracefully stop the training process"""
        if self.current_process and self.current_process.poll() is None:
            self.log("Sending graceful shutdown signal")
            self.current_process.terminate()

            # Wait for graceful shutdown
            try:
                self.current_process.wait(timeout=60)
                self.log("Training stopped gracefully")
            except subprocess.TimeoutExpired:
                self.log("Graceful shutdown timed out, forcing kill")
                self.current_process.kill()
                self.current_process.wait()

        self.is_running = False
        self.save_status("training_stopped")

    def schedule_restart(self):
        """Schedule a restart after cooldown period"""
        if self.restart_count >= self.max_restarts:
            self.log(f"Maximum restart attempts reached ({self.max_restarts})")
            self.save_status("max_restarts_reached")
            return

        self.log(f"Scheduling restart in {self.cooldown_minutes} minutes")
        self.save_status("cooldown_period")

        time.sleep(self.cooldown_minutes * 60)

        self.restart_count += 1
        self.log(f"Restarting training session (attempt {self.restart_count})")
        self.start_training_session()

    def run_continuous_cycle(self):
        """Run the continuous improvement cycle"""
        self.log("Starting continuous improvement monitor")

        # Load previous state
        self.load_status()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            while self.restart_count < self.max_restarts:
                self.start_training_session()

                # If training ended normally, schedule a restart
                if not self.is_running:
                    self.schedule_restart()

        except KeyboardInterrupt:
            self.log("Received interrupt signal")
            self.stop_training_gracefully()

        self.log("Continuous improvement cycle ended")

    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.log(f"Received signal {signum}")
        self.stop_training_gracefully()
        sys.exit(0)

    def get_status_report(self):
        """Generate a status report"""
        try:
            with open(self.status_file, "r") as f:
                status_data = json.load(f)

            # Read recent log entries
            recent_logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    lines = f.readlines()
                    recent_logs = lines[-20:]  # Last 20 log entries

            # Check for recent improvements
            improvements = []
            exp_dir = Path("experiments")
            if exp_dir.exists():
                for exp_path in sorted(exp_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if exp_path.is_dir():
                        improvements.append({
                            "experiment": exp_path.name,
                            "timestamp": datetime.fromtimestamp(exp_path.stat().st_mtime).isoformat()
                        })
                        if len(improvements) >= 5:
                            break

            report = {
                "status": status_data,
                "recent_logs": recent_logs,
                "recent_improvements": improvements,
                "system_info": {
                    "uptime": str(datetime.now() - datetime.fromisoformat(status_data.get("timestamp", datetime.now().isoformat()))),
                    "restart_count": self.restart_count
                }
            }

            return report

        except Exception as e:
            return {"error": f"Failed to generate report: {str(e)}"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MiniGPT Training Monitor")
    parser.add_argument("--action", choices=["start", "stop", "status", "restart"],
                       default="start", help="Action to perform")
    parser.add_argument("--session-hours", type=int, default=8,
                       help="Training session duration in hours")
    parser.add_argument("--max-restarts", type=int, default=5,
                       help="Maximum number of restart attempts")

    args = parser.parse_args()

    monitor = AutoTrainingMonitor()
    monitor.session_duration_hours = args.session_hours
    monitor.max_restarts = args.max_restarts

    if args.action == "start":
        monitor.run_continuous_cycle()
    elif args.action == "status":
        report = monitor.get_status_report()
        print(json.dumps(report, indent=2))
    elif args.action == "stop":
        monitor.log("Received stop command")
        monitor.stop_training_gracefully()
    elif args.action == "restart":
        monitor.restart_count = 0  # Reset restart count
        monitor.run_continuous_cycle()


if __name__ == "__main__":
    main()