#!/usr/bin/env python3
"""
Task Manager - A simple command-line task management application
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class TaskManager:
    """Manages tasks with persistence to JSON file"""
    
    def __init__(self, data_file: str = "tasks.json"):
        self.data_file = data_file
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_tasks(self):
        """Save tasks to JSON file"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, indent=2, ensure_ascii=False)
    
    def add_task(self, description: str, priority: str = "medium") -> int:
        """Add a new task"""
        task_id = len(self.tasks) + 1
        task = {
            "id": task_id,
            "description": description,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        self.tasks.append(task)
        self._save_tasks()
        return task_id
    
    def list_tasks(self, status: Optional[str] = None):
        """List all tasks, optionally filtered by status"""
        filtered_tasks = self.tasks
        if status:
            filtered_tasks = [t for t in self.tasks if t["status"] == status]
        
        if not filtered_tasks:
            print("Nu există task-uri.")
            return
        
        print("\n" + "="*60)
        print(f"{'ID':<5} {'Status':<12} {'Priority':<10} {'Description'}")
        print("="*60)
        
        for task in filtered_tasks:
            status_icon = "✓" if task["status"] == "completed" else "○"
            priority_icon = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(task["priority"], "⚪")
            
            print(f"{task['id']:<5} {status_icon} {task['status']:<10} "
                  f"{priority_icon} {task['priority']:<8} {task['description']}")
        
        print("="*60 + "\n")
    
    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                self._save_tasks()
                return True
        return False
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task"""
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                self.tasks.pop(i)
                self._save_tasks()
                return True
        return False
    
    def get_statistics(self):
        """Get task statistics"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t["status"] == "completed")
        pending = total - completed
        
        print("\n" + "="*40)
        print("Statistici Task-uri")
        print("="*40)
        print(f"Total: {total}")
        print(f"Finalizate: {completed}")
        print(f"În așteptare: {pending}")
        if total > 0:
            progress = (completed / total) * 100
            print(f"Progres: {progress:.1f}%")
        print("="*40 + "\n")


def main():
    """Main CLI interface"""
    manager = TaskManager()
    
    print("="*60)
    print("📋 Task Manager - Gestionare Task-uri")
    print("="*60)
    print("\nComenzi disponibile:")
    print("  add <descriere> [priority]  - Adaugă un task nou")
    print("  list [status]               - Listează task-urile")
    print("  complete <id>               - Marchează un task ca finalizat")
    print("  delete <id>                 - Șterge un task")
    print("  stats                       - Afișează statistici")
    print("  help                        - Afișează acest mesaj")
    print("  quit                        - Ieșire")
    print()
    
    while True:
        try:
            command = input("> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == "quit" or cmd == "exit" or cmd == "q":
                print("La revedere! 👋")
                break
            
            elif cmd == "help" or cmd == "h":
                print("\nComenzi disponibile:")
                print("  add <descriere> [high|medium|low]")
                print("  list [pending|completed]")
                print("  complete <id>")
                print("  delete <id>")
                print("  stats")
                print("  quit\n")
            
            elif cmd == "add" or cmd == "a":
                if len(command) < 2:
                    print("❌ Eroare: Specifică o descriere pentru task.")
                    continue
                
                description = " ".join(command[1:-1] if len(command) > 2 and command[-1] in ["high", "medium", "low"] else command[1:])
                priority = command[-1] if len(command) > 2 and command[-1] in ["high", "medium", "low"] else "medium"
                
                task_id = manager.add_task(description, priority)
                print(f"✅ Task adăugat cu succes! ID: {task_id}")
            
            elif cmd == "list" or cmd == "l":
                status = command[1] if len(command) > 1 else None
                manager.list_tasks(status)
            
            elif cmd == "complete" or cmd == "c":
                if len(command) < 2:
                    print("❌ Eroare: Specifică ID-ul task-ului.")
                    continue
                
                try:
                    task_id = int(command[1])
                    if manager.complete_task(task_id):
                        print(f"✅ Task {task_id} marcat ca finalizat!")
                    else:
                        print(f"❌ Task {task_id} nu a fost găsit.")
                except ValueError:
                    print("❌ Eroare: ID-ul trebuie să fie un număr.")
            
            elif cmd == "delete" or cmd == "d":
                if len(command) < 2:
                    print("❌ Eroare: Specifică ID-ul task-ului.")
                    continue
                
                try:
                    task_id = int(command[1])
                    if manager.delete_task(task_id):
                        print(f"✅ Task {task_id} șters cu succes!")
                    else:
                        print(f"❌ Task {task_id} nu a fost găsit.")
                except ValueError:
                    print("❌ Eroare: ID-ul trebuie să fie un număr.")
            
            elif cmd == "stats" or cmd == "s":
                manager.get_statistics()
            
            else:
                print(f"❌ Comandă necunoscută: {cmd}. Scrie 'help' pentru ajutor.")
        
        except KeyboardInterrupt:
            print("\n\nLa revedere! 👋")
            break
        except Exception as e:
            print(f"❌ Eroare: {e}")


if __name__ == "__main__":
    main()

