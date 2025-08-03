# Auto-Save Research Output System

## Overview

The Multi-Agent Research System now automatically saves research outputs in **three formats every time** a research mission is completed:

1. **📄 JSON** - Complete research data with all metadata
2. **📝 Markdown** - Professional formatted report 
3. **📋 Text** - Plain text version for maximum compatibility

## How It Works

### Automatic Saving

Every research mission automatically generates and saves all three formats when `save_outputs=True` (which is the default):

```python
from agents.multi_agent_orchestrator import MultiAgentResearchOrchestrator

# Initialize orchestrator
orchestrator = MultiAgentResearchOrchestrator(
    headless=True,
    output_dir="research_outputs"
)

# Run research mission - automatically saves JSON + Markdown + Text
results = await orchestrator.execute_research_mission(
    query="Your research question",
    research_depth="medium",
    report_type="comprehensive",
    save_outputs=True  # This is the default - saves all formats
)
```

### File Naming Convention

All files use a consistent naming pattern based on the mission ID:

- `mission_1752961918_complete.json` - Complete research data
- `mission_1752961918_report.md` - Markdown report
- `mission_1752961918_report.txt` - Text report
- `mission_1752961918_plan.json` - Research plan (optional)
- `mission_1752961918_research.json` - Raw research data (optional)

## Enhanced Features

### 🔄 Fallback Generation

If the formatted outputs are missing from the JSON data, the system automatically generates fallback versions from the report sections:

- Creates markdown structure with proper headings
- Generates plain text with section separators
- Preserves all available content and metadata

### 📊 File Tracking

The system provides detailed output summaries:

```
📁 Output Summary:
   • JSON: mission_1752961918_complete.json (245.7KB)
   • Markdown: mission_1752961918_report.md (23.4KB)
   • Text: mission_1752961918_report.txt (21.8KB)
   • Research Plan: mission_1752961918_plan.json (12.3KB)
   • Research Data: mission_1752961918_research.json (189.2KB)
```

### 🛠️ Extraction Utility

For existing JSON files that don't have markdown/text versions, use the extraction utility:

#### Single File Extraction

```bash
# Extract from a single JSON file
python src/utils/extract_outputs.py research_outputs/mission_1752961918_complete.json

# Specify custom output directory
python src/utils/extract_outputs.py research_outputs/mission_1752961918_complete.json -o my_outputs/
```

#### Batch Processing

```bash
# Process all JSON files in a directory
python src/utils/extract_outputs.py research_outputs/ --batch

# Process directory with custom output location
python src/utils/extract_outputs.py research_outputs/ --batch -o extracted_outputs/
```

#### Programmatic Extraction

```python
# Extract from existing JSON file programmatically
orchestrator = MultiAgentResearchOrchestrator()

# Extract from single file
success = await orchestrator.extract_formatted_outputs_from_json(
    "research_outputs/mission_1752961918_complete.json"
)

# List all saved missions with their available formats
missions = orchestrator.list_saved_research_missions()
for mission in missions:
    print(f"Mission: {mission['mission_id']}")
    print(f"  JSON: ✅" if mission['json_file'] else "  JSON: ❌")
    print(f"  Markdown: ✅" if mission['markdown_file'] else "  Markdown: ❌")
    print(f"  Text: ✅" if mission['text_file'] else "  Text: ❌")
```

## Output Formats

### 📄 JSON Format
Complete structured data including:
- Mission metadata and configuration
- Research plan details
- Source analysis and quality metrics
- Report sections and appendices
- Quality assessments and timing
- Raw formatted outputs

### 📝 Markdown Format
Professional report with:
```markdown
# Research Report: [Query]
**Report ID:** mission_xxx
**Generated:** timestamp
**Sources Analyzed:** N

## Executive Summary
[Content]

## Introduction
[Content]

## Methodology
[Content]

## Findings
[Content]

## Analysis
[Content]

## Recommendations
[Content]

## Source Bibliography
[Sources with credibility scores]
```

### 📋 Text Format
Plain text with section separators:
```
RESEARCH REPORT: [QUERY]

Report ID: mission_xxx
Generated: timestamp
Sources Analyzed: N

================================================================================

EXECUTIVE SUMMARY

[Content]

================================================================================

INTRODUCTION

[Content]

[... additional sections ...]
```

## System Integration

### Enhanced Orchestrator Methods

The `MultiAgentResearchOrchestrator` class now includes:

- `ensure_auto_save_enabled()` - Confirms auto-save is active
- `extract_formatted_outputs_from_json(path)` - Extract from existing JSON
- `list_saved_research_missions()` - List all saved missions
- Enhanced `_save_mission_outputs()` with fallback generation
- File size tracking and output summaries

### Error Handling

The system includes robust error handling:

- **Missing formatted outputs**: Automatically generates fallbacks
- **File write errors**: Creates backup files
- **Corrupted JSON**: Attempts recovery with available data
- **Directory issues**: Creates necessary directories

## Demo Scripts

### 🚀 Full Auto-Save Demo

Run a complete demonstration of the auto-save system:

```bash
python src/demo_auto_save.py
```

This will:
1. Run a sample research mission
2. Show all three formats being saved automatically
3. Demonstrate the extraction utility
4. List all saved missions

### 🔧 Manual Extraction

Test the extraction utility on existing files:

```bash
# Extract from your specific JSON file
python src/utils/extract_outputs.py research_outputs/mission_1752961918_complete.json

# Or process all files in the directory
python src/utils/extract_outputs.py research_outputs/ --batch
```

## Best Practices

1. **Always keep auto-save enabled**: `save_outputs=True` (default)
2. **Use consistent output directories**: Helps with organization
3. **Regular cleanup**: Archive old missions to prevent directory bloat  
4. **Backup important missions**: Copy complete JSON files for archival
5. **Use extraction utility**: For legacy JSON files missing formatted outputs

## Troubleshooting

### Missing Formatted Outputs

If markdown/text files aren't generated:

1. **Check the JSON**: Look for `formatted_outputs` section in the JSON file
2. **Use extraction utility**: `python src/utils/extract_outputs.py [json_file]`
3. **Check permissions**: Ensure write access to output directory
4. **Verify sections**: Ensure the report has content in the sections

### File Size Issues

For very large outputs:

- JSON files can be 100KB-1MB+ (complete data)
- Markdown files are typically 10-50KB (formatted report)
- Text files are similar size to Markdown

### Directory Structure

Recommended organization:
```
project/
├── research_outputs/           # Main output directory
│   ├── mission_xxx_complete.json
│   ├── mission_xxx_report.md
│   ├── mission_xxx_report.txt
│   └── ...
├── archived_research/          # Archive old missions
└── extracted_outputs/          # Extraction results
```

## Migration

If you have existing JSON research files without markdown/text outputs:

1. **Identify JSON files**: Look for `*_complete.json` files
2. **Run batch extraction**: `python src/utils/extract_outputs.py [directory] --batch`
3. **Verify outputs**: Check that markdown and text files were created
4. **Update workflow**: Ensure future missions use auto-save enabled

---

**🎯 Result**: Every research mission now automatically produces JSON + Markdown + Text outputs, ensuring maximum compatibility and usability across different platforms and use cases!
