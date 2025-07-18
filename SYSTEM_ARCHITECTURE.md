# PCB Auto-Inspection System - Architecture Diagrams

## Version 1.0 | December 2024

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Threading Architecture](#threading-architecture)
5. [Database Schema](#database-schema)
6. [Deployment Architecture](#deployment-architecture)
7. [Network Architecture](#network-architecture)
8. [Security Architecture](#security-architecture)

---

## System Overview

### High-Level System Architecture

```
                    PCB AUTO-INSPECTION SYSTEM
                           ARCHITECTURE
                              
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
│  │   Main GUI  │  │ Analytics   │  │   History   │  │ Config  │  │
│  │   Control   │  │   Viewer    │  │   Browser   │  │ Manager │  │
│  │   Panel     │  │             │  │             │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LOGIC LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
│  │ Main System │  │ Auto-Trigger│  │ Thread Pool │  │ Event   │  │
│  │Orchestrator │  │   Engine    │  │  Manager    │  │ Manager │  │
│  │             │  │             │  │             │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
│  │    Image    │  │  PCB & Focus│  │ AI Defect   │  │ Results │  │
│  │ Processing  │  │  Detection  │  │ Detection   │  │Processing│  │
│  │             │  │             │  │             │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA ACCESS LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
│  │  Database   │  │  Analytics  │  │ Report Gen  │  │ Backup  │  │
│  │  Manager    │  │   Engine    │  │   Engine    │  │ Manager │  │
│  │             │  │             │  │             │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HARDWARE ACCESS LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
│  │   Camera    │  │   Lighting  │  │   Storage   │  │ Network │  │
│  │ Controller  │  │  Controller │  │  Interface  │  │Interface│  │
│  │             │  │             │  │             │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PHYSICAL HARDWARE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │   Basler    │    │   LED Ring  │    │   Computer  │         │
│   │   Camera    │    │   Lighting  │    │   Hardware  │         │
│   │acA3800-10gm │    │    Array    │    │             │         │
│   └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Detailed Component Interaction

```
                     COMPONENT INTERACTION DIAGRAM
                             
┌─────────────────────────────────────────────────────────────────┐
│                         GUI LAYER                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Main Window                              ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   ││
│  │  │   Control   │ │   Preview   │ │     Results         │   ││
│  │  │   Panel     │ │   Display   │ │     Display         │   ││
│  │  │             │ │             │ │                     │   ││
│  │  │ [AUTO]      │ │   Camera    │ │  Defect List       │   ││
│  │  │ [Manual]    │ │   Stream    │ │  Bounding Boxes    │   ││
│  │  │ [Analytics] │ │   PCB Det.  │ │  Confidence        │   ││
│  │  │ [History]   │ │   Focus     │ │  Processing Time   │   ││
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
   ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
   │  Analytics GUI  │ │ History GUI │ │ Configuration   │
   │                 │ │             │ │     GUI         │
   │  • Charts       │ │ • Search    │ │ • Camera Setup  │
   │  • Trends       │ │ • Filter    │ │ • AI Settings   │
   │  • Reports      │ │ • Export    │ │ • Calibration   │
   └─────────────────┘ └─────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN SYSTEM CONTROLLER                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             PCBInspectionSystem                         │   │
│  │                                                         │   │
│  │  • Auto-trigger management                              │   │
│  │  • Preview stream coordination                          │   │
│  │  • Inspection workflow orchestration                    │   │
│  │  • GUI event handling                                   │   │
│  │  • Error recovery                                       │   │
│  │  • Threading coordination                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  PROCESSING     │ │    AI LAYER     │ │   DATA LAYER    │
│     LAYER       │ │                 │ │                 │
│                 │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ ┌─────────────┐ │ │ │ YOLOv11     │ │ │ │ SQLite DB   │ │
│ │Preprocessor │ │ │ │ Detector    │ │ │ │ Manager     │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │PCB Detector │ │ │ │ Class       │ │ │ │ Analytics   │ │
│ └─────────────┘ │ │ │ Mapping     │ │ │ │ Engine      │ │
│ ┌─────────────┐ │ │ └─────────────┘ │ │ └─────────────┘ │
│ │Postprocess  │ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ └─────────────┘ │ │ │Performance  │ │ │ │ Report      │ │
│                 │ │ │ Monitor     │ │ │ │ Generator   │ │
└─────────────────┘ │ └─────────────┘ │ │ └─────────────┘ │
                    └─────────────────┘ └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ HARDWARE LAYER  │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ Basler      │ │
                    │ │ Camera      │ │
                    │ │ Controller  │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ Camera      │ │
                    │ │ Presets     │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │ Image       │ │
                    │ │ Handler     │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

---

## Data Flow Diagrams

### Auto-Trigger Inspection Flow

```
                         AUTO-TRIGGER INSPECTION FLOW
                                     
Start ┌─────────────────────────────────────────────────────────┐
      │                                                         │
      ▼                                                         │
┌─────────────────┐        ┌─────────────────┐                 │
│ Camera Preview  │───────▶│ PCB Detection   │                 │
│ Stream Started  │        │ & Focus Check   │                 │
│                 │        │                 │                 │
│ • 30 FPS        │        │ • Edge detect   │                 │
│ • Low exposure  │        │ • Contour find  │                 │
│ • Queue mgmt    │        │ • Focus score   │                 │
└─────────────────┘        └─────────────────┘                 │
                                     │                         │
                                     ▼                         │
                          ┌─────────────────┐                 │
                          │ Stability Check │                 │
                          │                 │                 │
                          │ • Position      │                 │
                          │ • 10 frames     │                 │
                          │ • Movement tol  │                 │
                          └─────────────────┘                 │
                                     │                         │
                                     ▼                         │
                             [PCB Stable?]                     │
                                     │                         │
                            No ──────┘                         │
                                     │ Yes                     │
                                     ▼                         │
                          ┌─────────────────┐                 │
                          │ Focus Quality   │                 │
                          │ Assessment      │                 │
                          │                 │                 │
                          │ • Laplacian var │                 │
                          │ • Threshold     │                 │
                          │ • Score > 100   │                 │
                          └─────────────────┘                 │
                                     │                         │
                                     ▼                         │
                             [Focus OK?]                       │
                                     │                         │
                            No ──────┘                         │
                                     │ Yes                     │
                                     ▼                         │
                        ┌─────────────────────┐               │
                        │ Inspection Interval │               │
                        │ Check               │               │
                        │                     │               │
                        │ • Last inspect time │               │
                        │ • Min 2 seconds     │               │
                        │ • Rate limiting     │               │
                        └─────────────────────┘               │
                                     │                         │
                                     ▼                         │
                            [Interval OK?]                     │
                                     │                         │
                            No ──────┘                         │
                                     │ Yes                     │
                                     ▼                         │
                         ┌─────────────────────┐              │
                         │ TRIGGER INSPECTION  │              │
                         │                     │              │
                         │ • High-quality cap  │              │
                         │ • AI inference      │              │
                         │ • Result processing │              │
                         └─────────────────────┘              │
                                     │                         │
                                     ▼                         │
                                Continue ─────────────────────┘
```

### High-Quality Inspection Pipeline

```
                      HIGH-QUALITY INSPECTION PIPELINE
                                     
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│    Trigger      │───────▶│ High-Quality    │───────▶│ Raw to Processed│
│   Inspection    │        │ Image Capture   │        │ Image Conversion│
│                 │        │                 │        │                 │
│ • Auto/Manual   │        │ • Stop preview  │        │ • Bayer→Gray    │
│ • Thread safe   │        │ • High exposure │        │ • CLAHE enhance │
│ • Rate limited  │        │ • Single frame  │        │ • Bilateral flt │
└─────────────────┘        └─────────────────┘        └─────────────────┘
                                     │                         │
                                     ▼                         ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Database Save  │◄───────│ Results Extract │◄───────│ AI Inference    │
│                 │        │                 │        │                 │
│ • Metadata only │        │ • Class mapping │        │ • YOLOv11 model │
│ • Defect images │        │ • Bbox extract  │        │ • GPU inference │
│ • Statistics    │        │ • Confidence    │        │ • FP16 precision│
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                           │                         │
         ▼                           ▼                         ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Analytics Update│        │ GUI Results     │        │ Performance Log │
│                 │        │ Display         │        │                 │
│ • Real-time     │        │                 │        │ • Timing stats  │
│ • Cached stats  │        │ • Defect visual │        │ • Memory usage  │
│ • Trends        │        │ • Result text   │        │ • Throughput    │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

---

## Threading Architecture

### Thread Model and Communication

```
                           THREADING ARCHITECTURE
                                    
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 GUI Main Loop                           │   │
│  │                                                         │   │
│  │  • tkinter main loop                                    │   │
│  │  • Event handling                                       │   │
│  │  • Display updates                                      │   │
│  │  • User interaction                                     │   │
│  │  • System coordination                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                 THREAD POOL                             │
    │                                                         │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
    │  │   Preview   │ │ Inspection  │ │   Database      │   │
    │  │   Thread    │ │   Thread    │ │   Thread        │   │
    │  │             │ │             │ │                 │   │
    │  │ • Camera    │ │ • AI Inf.   │ │ • Async writes  │   │
    │  │   stream    │ │ • High qual │ │ • Bulk insert   │   │
    │  │ • PCB det.  │ │ • Processing│ │ • Index maint   │   │
    │  │ • GUI upd.  │ │ • Results   │ │ • Backup        │   │
    │  └─────────────┘ └─────────────┘ └─────────────────┘   │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
           ┌─────────────────────────────────────────┐
           │         SYNCHRONIZATION                 │
           │                                         │
           │  ┌─────────────┐ ┌─────────────────┐   │
           │  │   Queues    │ │     Locks       │   │
           │  │             │ │                 │   │
           │  │ • Frame     │ │ • Inspection    │   │
           │  │   buffer    │ │   lock          │   │
           │  │ • Result    │ │ • Database      │   │
           │  │   queue     │ │   lock          │   │
           │  │ • Error     │ │ • GUI update    │   │
           │  │   queue     │ │   lock          │   │
           │  └─────────────┘ └─────────────────┘   │
           └─────────────────────────────────────────┘
```

### Thread Communication Pattern

```
                    THREAD COMMUNICATION PATTERN
                             
┌─────────────────┐         ┌─────────────────┐
│   Preview       │         │   Inspection    │
│   Thread        │         │   Thread        │
│                 │         │                 │
│ 1. Grab frame   │         │ 1. Wait trigger │
│ 2. PCB detect   │         │ 2. Capture HQ   │
│ 3. Focus check  │         │ 3. Process img  │
│ 4. Stability    │         │ 4. AI inference │
│ 5. Auto trigger │────────▶│ 5. Extract res  │
│ 6. GUI update   │         │ 6. Save to DB   │
│                 │         │ 7. Update GUI   │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Frame Queue   │         │  Result Queue   │
│                 │         │                 │
│ • Size: 10      │         │ • Inspection    │
│ • Drop oldest   │         │   results       │
│ • Thread safe   │         │ • Statistics    │
│ • Non-blocking  │         │ • Error info    │
└─────────────────┘         └─────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
                ┌─────────────────┐
                │   Main Thread   │
                │                 │
                │ • GUI updates   │
                │ • Event handle  │
                │ • Coordination  │
                └─────────────────┘
```

---

## Database Schema

### Complete Database Schema Diagram

```
                           DATABASE SCHEMA
                                
┌─────────────────────────────────────────────────────────────────┐
│                         INSPECTIONS TABLE                       │
├─────────────────────────────────────────────────────────────────┤
│ id                     INTEGER PRIMARY KEY AUTOINCREMENT       │
│ timestamp              TEXT NOT NULL                            │
│ unix_timestamp         REAL NOT NULL                            │
│ has_defects           BOOLEAN NOT NULL                          │
│ defect_count          INTEGER NOT NULL                          │
│ defects               TEXT (JSON)                               │
│ defect_locations      TEXT (JSON)                               │
│ confidence_scores     TEXT (JSON)                               │
│ focus_score           REAL                                      │
│ processing_time       REAL                                      │
│ image_path           TEXT                                       │
│ pcb_area             INTEGER                                    │
│ trigger_type         TEXT                                       │
│ session_id           TEXT                                       │
├─────────────────────────────────────────────────────────────────┤
│ INDEXES:                                                        │
│ • idx_timestamp        ON (timestamp)                          │
│ • idx_unix_timestamp   ON (unix_timestamp)                     │
│ • idx_has_defects      ON (has_defects)                        │
│ • idx_session_id       ON (session_id)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEFECT_STATISTICS TABLE                      │
├─────────────────────────────────────────────────────────────────┤
│ defect_type           TEXT PRIMARY KEY                          │
│ total_count           INTEGER DEFAULT 0                         │
│ last_seen             TEXT                                      │
│ first_seen            TEXT                                      │
│ avg_confidence        REAL                                      │
│ trend_direction       TEXT                                      │
├─────────────────────────────────────────────────────────────────┤
│ INDEXES:                                                        │
│ • idx_total_count      ON (total_count DESC)                   │
│ • idx_last_seen        ON (last_seen)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE_METRICS TABLE                    │
├─────────────────────────────────────────────────────────────────┤
│ id                    INTEGER PRIMARY KEY AUTOINCREMENT        │
│ timestamp             TEXT NOT NULL                             │
│ metric_type           TEXT NOT NULL                             │
│ metric_value          REAL NOT NULL                             │
│ session_id            TEXT                                      │
│ component             TEXT                                      │
├─────────────────────────────────────────────────────────────────┤
│ INDEXES:                                                        │
│ • idx_perf_timestamp   ON (timestamp)                          │
│ • idx_perf_type        ON (metric_type)                        │
│ • idx_perf_session     ON (session_id)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM_EVENTS TABLE                        │
├─────────────────────────────────────────────────────────────────┤
│ id                    INTEGER PRIMARY KEY AUTOINCREMENT        │
│ timestamp             TEXT NOT NULL                             │
│ event_type            TEXT NOT NULL                             │
│ event_level           TEXT NOT NULL                             │
│ message               TEXT                                      │
│ component             TEXT                                      │
│ session_id            TEXT                                      │
│ stack_trace           TEXT                                      │
├─────────────────────────────────────────────────────────────────┤
│ INDEXES:                                                        │
│ • idx_event_timestamp  ON (timestamp)                          │
│ • idx_event_type       ON (event_type)                         │
│ • idx_event_level      ON (event_level)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Relationships

```
                         DATA RELATIONSHIP DIAGRAM
                                  
┌─────────────────┐        ┌─────────────────┐
│   INSPECTIONS   │        │ DEFECT_STATS    │
│                 │        │                 │
│ • Master record │───────▶│ • Aggregated    │
│ • All metadata  │  1:N   │ • Counts        │
│ • JSON defects  │        │ • Trends        │
└─────────────────┘        └─────────────────┘
         │                           │
         │ 1:N                      │
         ▼                           ▼
┌─────────────────┐        ┌─────────────────┐
│ PERFORMANCE     │        │ SYSTEM_EVENTS   │
│                 │        │                 │
│ • Timing data   │        │ • Logs          │
│ • Memory usage  │        │ • Errors        │
│ • Throughput    │        │ • Warnings      │
└─────────────────┘        └─────────────────┘
```

---

## Deployment Architecture

### Production Deployment

```
                        PRODUCTION DEPLOYMENT
                               
┌─────────────────────────────────────────────────────────────────┐
│                      PRODUCTION ENVIRONMENT                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  MAIN APPLICATION                       │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   GUI App   │  │  Database   │  │ File System │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Main UI   │  │ • SQLite    │  │ • Images    │    │   │
│  │  │ • Analytics │  │ • WAL mode  │  │ • Logs      │    │   │
│  │  │ • History   │  │ • Indexes   │  │ • Backups   │    │   │
│  │  │ • Config    │  │ • Stats     │  │ • Reports   │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  HARDWARE INTERFACES                    │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Camera    │  │  Lighting   │  │   Network   │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • USB 3.0   │  │ • LED Ring  │  │ • Ethernet  │    │   │
│  │  │ • Basler    │  │ • Control   │  │ • Remote    │    │   │
│  │  │ • pypylon   │  │ • PWM       │  │ • Updates   │    │   │
│  │  │ • Drivers   │  │ • Sensors   │  │ • Support   │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM INFRASTRUCTURE                       │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │   Hardware  │  │ Operating   │  │   Runtime   │  │Security ││
│  │   Platform  │  │   System    │  │ Environment │  │ System  ││
│  │             │  │             │  │             │  │         ││
│  │ • Intel i5+ │  │ • Windows   │  │ • Python    │  │ • User  ││
│  │ • 16GB RAM  │  │   10/11     │  │   3.8+      │  │   mgmt  ││
│  │ • Tesla P4  │  │ • Ubuntu    │  │ • CUDA      │  │ • Access││
│  │ • SSD       │  │   22.04     │  │ • OpenCV    │  │ • Audit ││
│  │ • USB 3.0   │  │ • Drivers   │  │ • PyTorch   │  │ • Logs  ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Development/Test Environment

```
                    DEVELOPMENT/TEST ENVIRONMENT
                              
┌─────────────────────────────────────────────────────────────────┐
│                      DEVELOPMENT SETUP                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SOURCE CODE                          │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Git Repo  │  │    Tests    │  │    Docs     │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Main code │  │ • Unit      │  │ • API docs  │    │   │
│  │  │ • Branches  │  │ • Integration│  │ • User man  │    │   │
│  │  │ • History   │  │ • Performance│  │ • Architecture │  │   │
│  │  │ • Tags      │  │ • Mocks     │  │ • Diagrams  │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   MOCK ENVIRONMENT                      │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ Mock Camera │  │  Mock AI    │  │  Test Data  │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Synthetic │  │ • Simulated │  │ • Sample    │    │   │
│  │  │   images    │  │   results   │  │   images    │    │   │
│  │  │ • pypylon   │  │ • Fast      │  │ • Test DB   │    │   │
│  │  │   mock      │  │   inference │  │ • Fixtures  │    │   │
│  │  │ • Error sim │  │ • YOLO mock │  │ • Scenarios │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Network Architecture

### System Network Integration

```
                         NETWORK ARCHITECTURE
                               
┌─────────────────────────────────────────────────────────────────┐
│                      ENTERPRISE NETWORK                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  EXTERNAL SERVICES                      │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Remote    │  │   Update    │  │   Support   │    │   │
│  │  │ Monitoring  │  │   Server    │  │   Portal    │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Telemetry │  │ • Software  │  │ • Remote    │    │   │
│  │  │ • Alerts    │  │ • Models    │  │   access    │    │   │
│  │  │ • Analytics │  │ • Config    │  │ • Diagnostics│   │   │
│  │  │ • Reports   │  │ • Patches   │  │ • Training  │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FACTORY NETWORK                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  NETWORK SERVICES                       │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │    DHCP     │  │     DNS     │  │    NTP      │    │   │
│  │  │   Server    │  │   Server    │  │   Server    │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • IP assign │  │ • Name res  │  │ • Time sync │    │   │
│  │  │ • Leases    │  │ • Local DNS │  │ • Precision │    │   │
│  │  │ • VLAN      │  │ • Caching   │  │ • Accuracy  │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INSPECTION STATION                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 LOCAL INTERFACES                        │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Camera    │  │  Computer   │  │   Display   │    │   │
│  │  │ Interface   │  │  Network    │  │  Interface  │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • USB 3.0   │  │ • Ethernet  │  │ • Monitor   │    │   │
│  │  │ • Direct    │  │ • TCP/IP    │  │ • Keyboard  │    │   │
│  │  │ • High BW   │  │ • Security  │  │ • Mouse     │    │   │
│  │  │ • Real-time │  │ • Firewall  │  │ • Touch     │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Security Model and Controls

```
                          SECURITY ARCHITECTURE
                               
┌─────────────────────────────────────────────────────────────────┐
│                       SECURITY LAYERS                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                APPLICATION SECURITY                     │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │    User     │  │    Data     │  │   System    │    │   │
│  │  │   Access    │  │ Protection  │  │ Integrity   │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Login     │  │ • Encryption│  │ • Code sign │    │   │
│  │  │ • Roles     │  │ • Backups   │  │ • Checksums │    │   │
│  │  │ • Audit     │  │ • Retention │  │ • Updates   │    │   │
│  │  │ • Session   │  │ • Sanitize  │  │ • Validation│    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 NETWORK SECURITY                        │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │  Firewall   │  │    VPN      │  │ Intrusion   │    │   │
│  │  │ Protection  │  │   Access    │  │ Detection   │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Inbound   │  │ • Encrypted │  │ • Monitor   │    │   │
│  │  │ • Outbound  │  │ • Auth      │  │ • Alerting  │    │   │
│  │  │ • Ports     │  │ • Tunnels   │  │ • Response  │    │   │
│  │  │ • Protocols │  │ • Certs     │  │ • Forensics │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 PHYSICAL SECURITY                       │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Device    │  │   Access    │  │ Environment │    │   │
│  │  │ Protection  │  │  Control    │  │ Protection  │    │   │
│  │  │             │  │             │  │             │    │   │
│  │  │ • Hardware  │  │ • Badge     │  │ • Climate   │    │   │
│  │  │ • Tamper    │  │ • Biometric │  │ • Power     │    │   │
│  │  │ • Locks     │  │ • Logs      │  │ • Surge     │    │   │
│  │  │ • Cables    │  │ • Zones     │  │ • Backup    │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Security Controls Matrix

```
                        SECURITY CONTROLS MATRIX
                              
                  ┌─────────┬─────────┬─────────┬─────────┐
                  │  USER   │  DATA   │ NETWORK │ SYSTEM  │
┌─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ AUTHENTICATION  │ ✓ LOGIN │ ✓ ENC   │ ✓ VPN   │ ✓ CERT  │
├─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ AUTHORIZATION   │ ✓ RBAC  │ ✓ ACL   │ ✓ FW    │ ✓ PERM  │
├─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ AUDITING        │ ✓ LOG   │ ✓ TRAIL │ ✓ MON   │ ✓ EVENT │
├─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ ENCRYPTION      │ ✓ PASS  │ ✓ AES   │ ✓ TLS   │ ✓ DISK  │
├─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ BACKUP          │ ✓ PROF  │ ✓ AUTO  │ ✓ REPL  │ ✓ IMAGE │
├─────────────────┼─────────┼─────────┼─────────┼─────────┤
│ MONITORING      │ ✓ SESS  │ ✓ ACCESS│ ✓ TRAF  │ ✓ PERF  │
└─────────────────┴─────────┴─────────┴─────────┴─────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Prepared by:** PCB Inspection System Development Team  

This document provides comprehensive architectural diagrams and technical specifications for the PCB Auto-Inspection System. For implementation details, refer to the API Documentation and source code.