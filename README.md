# VisionAware-Agent
An AI-powered visual perception agent with natural-language interaction and real-time object detection.

# Project Overview 🚀 

VisionAware Agent is an intelligent AI system that combines computer vision, LangChain, and LangGraph to understand and describe what it sees through a camera.
You can simply ask the agent questions like:

“What do you see?”

“How many objects are in the frame?”

And it responds clearly:

“I see 5 persons, 2 goats, and 3 chairs.”

The agent processes visual data using a custom YOLO computer vision pipeline built as a tool inside a LangChain–LangGraph workflow. It merges perception, reasoning and natural-language explanation into a single intelligent system.

# How It Works

1. User asks: “What do you see?”

2. The LangChain agent triggers the Vision Tool.

3. The custom YOLO pipeline:

Captures a frame

Runs YOLOv8 inference

Extracts object labels & counts

4. LangGraph processes the results

5. The agent produces a natural-language response with aggregated visual information.

# Tech Stack

1. Python

2. YOLOv8n (Ultralytics)

3. OpenCV

4. LangChain – tool creation & prompt interaction

5. LangGraph – reasoning flow and agent orchestration

6. Camera / Webcam for real-time feed

# Motivation 💡
This project is meaningful and personal, created from a desire to merge robotics, AI and computer vision into a single intelligent agent.
The goal was to build a system that sees, understands and responds naturally—like a small step toward real-world AI assistants.

# Developer

Babalola John Abidemi
Engineering student passionate about robotics, computer vision, embedded systems and agentic AI.
This project reflects my drive to build intelligent tools that support automation, learning, research and real-world problem solving.
