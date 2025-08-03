"""
Collaborative Research Features

Advanced collaborative research capabilities including team coordination,
knowledge sharing, distributed research, and multi-agent collaboration.
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from pathlib import Path
import hashlib


class CollaborationType(Enum):
    """Types of collaborative research"""

    DISTRIBUTED = "distributed"  # Multiple researchers on different aspects
    PARALLEL = "parallel"  # Multiple researchers on same topic
    SEQUENTIAL = "sequential"  # Handoff between researchers
    PEER_REVIEW = "peer_review"  # Review and validation by peers
    CROWDSOURCED = "crowdsourced"  # Community-driven research


class ResearchRole(Enum):
    """Roles in collaborative research"""

    LEAD_RESEARCHER = "lead_researcher"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


class TaskStatus(Enum):
    """Status of collaborative tasks"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_NEEDED = "revision_needed"


@dataclass
class Researcher:
    """Individual researcher in collaborative project"""

    id: str
    name: str
    role: ResearchRole
    specializations: List[str] = field(default_factory=list)
    availability: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)

    # Collaboration metrics
    collaboration_score: float = 1.0
    reliability_score: float = 1.0
    quality_score: float = 1.0
    communication_score: float = 1.0


@dataclass
class CollaborativeTask:
    """Task in collaborative research project"""

    id: str
    title: str
    description: str
    assigned_to: Optional[str] = None  # Researcher ID
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1  # 1 = highest, 5 = lowest

    # Task details
    research_area: str = ""
    expected_output: str = ""
    resources_needed: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other task IDs

    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    findings: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    confidence_score: float = 0.0

    # Collaboration
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    revisions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """Shared knowledge base for collaborative research"""

    id: str
    name: str
    description: str

    # Content
    documents: Dict[str, Any] = field(default_factory=dict)
    research_findings: Dict[str, Any] = field(default_factory=dict)
    source_library: Dict[str, Any] = field(default_factory=dict)
    methodologies: Dict[str, Any] = field(default_factory=dict)

    # Organization
    tags: Set[str] = field(default_factory=set)
    categories: Dict[str, List[str]] = field(default_factory=dict)

    # Access control
    contributors: Set[str] = field(default_factory=set)  # Researcher IDs
    access_level: str = "team"  # public, team, restricted

    # Versioning
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    update_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CollaborativeProject:
    """Collaborative research project"""

    id: str
    name: str
    description: str
    collaboration_type: CollaborationType

    # Team
    researchers: Dict[str, Researcher] = field(default_factory=dict)
    coordinator: Optional[str] = None  # Researcher ID

    # Research scope
    research_questions: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    scope: str = ""
    methodology: str = ""

    # Tasks and workflow
    tasks: Dict[str, CollaborativeTask] = field(default_factory=dict)
    workflow: Dict[str, Any] = field(default_factory=dict)
    milestones: List[Dict[str, Any]] = field(default_factory=list)

    # Knowledge management
    knowledge_base: Optional[KnowledgeBase] = None
    shared_resources: Dict[str, Any] = field(default_factory=dict)

    # Communication
    communication_channels: List[str] = field(default_factory=list)
    meeting_schedule: Dict[str, Any] = field(default_factory=dict)

    # Project timeline
    created_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    target_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None

    # Quality and progress
    overall_progress: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_metrics: Dict[str, float] = field(default_factory=dict)


class TaskCoordinator:
    """Coordinates task assignment and execution"""

    def __init__(self):
        self.assignment_algorithm = "skill_based"  # skill_based, load_balanced, random
        self.quality_threshold = 0.7
        self.review_required_threshold = 0.8  # Tasks above this quality need review

    def assign_task(self, task: CollaborativeTask, project: CollaborativeProject) -> Optional[str]:
        """Assign task to best available researcher"""

        if self.assignment_algorithm == "skill_based":
            return self._skill_based_assignment(task, project)
        elif self.assignment_algorithm == "load_balanced":
            return self._load_balanced_assignment(task, project)
        else:
            return self._random_assignment(task, project)

    def _skill_based_assignment(
        self, task: CollaborativeTask, project: CollaborativeProject
    ) -> Optional[str]:
        """Assign based on researcher skills and specializations"""

        best_researcher = None
        best_score = 0.0

        for researcher_id, researcher in project.researchers.items():
            # Skip if researcher is not available
            if not self._check_availability(researcher, task):
                continue

            # Calculate skill match score
            skill_score = self._calculate_skill_match(researcher, task)

            # Factor in performance metrics
            performance_score = (
                researcher.quality_score * 0.4
                + researcher.reliability_score * 0.3
                + researcher.collaboration_score * 0.2
                + researcher.communication_score * 0.1
            )

            # Factor in current workload (prefer less loaded researchers)
            workload_factor = max(0.1, 1.0 - (len(researcher.current_tasks) / 10))

            total_score = skill_score * 0.5 + performance_score * 0.3 + workload_factor * 0.2

            if total_score > best_score:
                best_score = total_score
                best_researcher = researcher_id

        return best_researcher

    def _load_balanced_assignment(
        self, task: CollaborativeTask, project: CollaborativeProject
    ) -> Optional[str]:
        """Assign to researcher with least current workload"""

        available_researchers = [
            (researcher_id, researcher)
            for researcher_id, researcher in project.researchers.items()
            if self._check_availability(researcher, task)
        ]

        if not available_researchers:
            return None

        # Sort by current workload (ascending)
        available_researchers.sort(key=lambda x: len(x[1].current_tasks))

        return available_researchers[0][0]

    def _random_assignment(
        self, task: CollaborativeTask, project: CollaborativeProject
    ) -> Optional[str]:
        """Random assignment to available researchers"""
        import random

        available_researchers = [
            researcher_id
            for researcher_id, researcher in project.researchers.items()
            if self._check_availability(researcher, task)
        ]

        return random.choice(available_researchers) if available_researchers else None

    def _check_availability(self, researcher: Researcher, task: CollaborativeTask) -> bool:
        """Check if researcher is available for task"""

        # Check basic availability
        if len(researcher.current_tasks) >= 5:  # Maximum concurrent tasks
            return False

        # Check role compatibility
        required_roles = {
            ResearchRole.SPECIALIST: ["specialist_task", "analysis"],
            ResearchRole.REVIEWER: ["review", "validation"],
            ResearchRole.SYNTHESIZER: ["synthesis", "integration"],
        }

        for role, task_types in required_roles.items():
            if researcher.role == role and not any(
                t in task.research_area.lower() for t in task_types
            ):
                return False

        return True

    def _calculate_skill_match(self, researcher: Researcher, task: CollaborativeTask) -> float:
        """Calculate how well researcher skills match task requirements"""

        if not researcher.specializations:
            return 0.5  # Default score if no specializations

        task_keywords = task.description.lower().split() + [task.research_area.lower()]

        matches = 0
        for specialization in researcher.specializations:
            if any(keyword in specialization.lower() for keyword in task_keywords):
                matches += 1

        return min(matches / len(researcher.specializations), 1.0)

    def reassign_task(self, task_id: str, project: CollaborativeProject, reason: str = "") -> bool:
        """Reassign task to different researcher"""

        if task_id not in project.tasks:
            return False

        task = project.tasks[task_id]

        # Remove from current researcher
        if task.assigned_to and task.assigned_to in project.researchers:
            current_researcher = project.researchers[task.assigned_to]
            if task_id in current_researcher.current_tasks:
                current_researcher.current_tasks.remove(task_id)

        # Reset task status
        task.status = TaskStatus.PENDING
        task.assigned_to = None
        task.assigned_at = None
        task.started_at = None

        # Add reassignment to revisions
        task.revisions.append(
            {
                "type": "reassignment",
                "reason": reason,
                "timestamp": datetime.now(),
                "previous_assignee": task.assigned_to,
            }
        )

        # Assign to new researcher
        new_assignee = self.assign_task(task, project)
        if new_assignee:
            task.assigned_to = new_assignee
            task.assigned_at = datetime.now()
            project.researchers[new_assignee].current_tasks.append(task_id)
            return True

        return False


class PeerReviewSystem:
    """Manages peer review process for collaborative research"""

    def __init__(self):
        self.min_reviewers = 2
        self.review_timeout_days = 7
        self.consensus_threshold = 0.8  # Agreement level required for approval

    def initiate_review(self, task: CollaborativeTask, project: CollaborativeProject) -> List[str]:
        """Initiate peer review for completed task"""

        # Select reviewers
        reviewers = self._select_reviewers(task, project)

        if len(reviewers) < self.min_reviewers:
            return []

        # Create review requests
        for reviewer_id in reviewers:
            review_request = {
                "reviewer_id": reviewer_id,
                "task_id": task.id,
                "requested_at": datetime.now(),
                "due_date": datetime.now() + timedelta(days=self.review_timeout_days),
                "status": "pending",
                "review_criteria": [
                    "accuracy",
                    "completeness",
                    "methodology",
                    "source_quality",
                    "clarity",
                    "relevance",
                ],
            }
            task.reviews.append(review_request)

        task.status = TaskStatus.UNDER_REVIEW
        return reviewers

    def _select_reviewers(
        self, task: CollaborativeTask, project: CollaborativeProject
    ) -> List[str]:
        """Select appropriate reviewers for task"""

        potential_reviewers = []

        for researcher_id, researcher in project.researchers.items():
            # Skip task assignee
            if researcher_id == task.assigned_to:
                continue

            # Check if researcher can review this type of task
            if self._can_review_task(researcher, task):
                potential_reviewers.append((researcher_id, researcher))

        # Sort by review capability score
        potential_reviewers.sort(
            key=lambda x: self._calculate_review_score(x[1], task), reverse=True
        )

        # Select top reviewers
        return [r[0] for r in potential_reviewers[: self.min_reviewers]]

    def _can_review_task(self, researcher: Researcher, task: CollaborativeTask) -> bool:
        """Check if researcher can review this task"""

        # Reviewers should have relevant expertise
        if researcher.role not in [
            ResearchRole.REVIEWER,
            ResearchRole.SPECIALIST,
            ResearchRole.LEAD_RESEARCHER,
        ]:
            return False

        # Check specialization overlap
        task_area = task.research_area.lower()
        return any(
            spec.lower() in task_area or task_area in spec.lower()
            for spec in researcher.specializations
        )

    def _calculate_review_score(self, researcher: Researcher, task: CollaborativeTask) -> float:
        """Calculate reviewer suitability score"""

        # Base score from performance metrics
        base_score = (
            researcher.quality_score * 0.4
            + researcher.reliability_score * 0.3
            + researcher.collaboration_score * 0.2
            + researcher.communication_score * 0.1
        )

        # Specialization bonus
        specialization_bonus = 0.0
        for spec in researcher.specializations:
            if spec.lower() in task.research_area.lower():
                specialization_bonus += 0.1

        return min(base_score + specialization_bonus, 1.0)

    def submit_review(
        self,
        task_id: str,
        reviewer_id: str,
        review_data: Dict[str, Any],
        project: CollaborativeProject,
    ) -> bool:
        """Submit review for task"""

        if task_id not in project.tasks:
            return False

        task = project.tasks[task_id]

        # Find review request
        review_request = None
        for review in task.reviews:
            if review["reviewer_id"] == reviewer_id and review["status"] == "pending":
                review_request = review
                break

        if not review_request:
            return False

        # Update review request
        review_request.update(
            {
                "status": "completed",
                "submitted_at": datetime.now(),
                "review_data": review_data,
                "scores": review_data.get("scores", {}),
                "comments": review_data.get("comments", ""),
                "recommendation": review_data.get(
                    "recommendation", "approve"
                ),  # approve, reject, revise
            }
        )

        # Check if all reviews are complete
        if self._all_reviews_complete(task):
            self._process_review_results(task)

        return True

    def _all_reviews_complete(self, task: CollaborativeTask) -> bool:
        """Check if all reviews for task are complete"""
        return all(review["status"] == "completed" for review in task.reviews)

    def _process_review_results(self, task: CollaborativeTask) -> None:
        """Process review results and make final decision"""

        if not task.reviews:
            return

        # Calculate consensus
        recommendations = [review["recommendation"] for review in task.reviews]
        approve_count = recommendations.count("approve")
        total_reviews = len(recommendations)

        consensus_score = approve_count / total_reviews

        # Aggregate scores
        all_scores = {}
        for review in task.reviews:
            scores = review.get("scores", {})
            for criterion, score in scores.items():
                if criterion not in all_scores:
                    all_scores[criterion] = []
                all_scores[criterion].append(score)

        # Calculate average scores
        avg_scores = {
            criterion: sum(scores) / len(scores) for criterion, scores in all_scores.items()
        }

        # Make final decision
        if consensus_score >= self.consensus_threshold:
            task.status = TaskStatus.APPROVED
            task.quality_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.7
        elif consensus_score < 0.3:  # Strong rejection
            task.status = TaskStatus.REJECTED
        else:
            task.status = TaskStatus.REVISION_NEEDED

        # Aggregate feedback
        all_feedback = []
        for review in task.reviews:
            if review.get("comments"):
                all_feedback.append(review["comments"])

        task.feedback = all_feedback


class KnowledgeManager:
    """Manages shared knowledge base for collaborative research"""

    def __init__(self):
        self.version_control = True
        self.auto_sync = True
        self.conflict_resolution = "merge"  # merge, overwrite, manual

    def create_knowledge_base(
        self, project: CollaborativeProject, name: str, description: str
    ) -> KnowledgeBase:
        """Create new knowledge base for project"""

        kb_id = f"kb_{project.id}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        knowledge_base = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            contributors=set(project.researchers.keys()),
            access_level="team",
        )

        project.knowledge_base = knowledge_base
        return knowledge_base

    def add_document(
        self, kb: KnowledgeBase, doc_id: str, content: Dict[str, Any], contributor_id: str
    ) -> bool:
        """Add document to knowledge base"""

        if contributor_id not in kb.contributors:
            return False

        # Create document entry
        document = {
            "id": doc_id,
            "content": content,
            "contributor": contributor_id,
            "created_at": datetime.now(),
            "last_modified": datetime.now(),
            "version": "1.0",
            "tags": content.get("tags", []),
            "category": content.get("category", "general"),
        }

        kb.documents[doc_id] = document

        # Update knowledge base metadata
        kb.last_updated = datetime.now()
        kb.update_history.append(
            {
                "action": "document_added",
                "document_id": doc_id,
                "contributor": contributor_id,
                "timestamp": datetime.now(),
            }
        )

        # Auto-categorize
        self._auto_categorize_document(kb, doc_id, document)

        return True

    def add_research_finding(
        self, kb: KnowledgeBase, finding_id: str, finding_data: Dict[str, Any], contributor_id: str
    ) -> bool:
        """Add research finding to knowledge base"""

        finding = {
            "id": finding_id,
            "data": finding_data,
            "contributor": contributor_id,
            "confidence_score": finding_data.get("confidence", 0.5),
            "sources": finding_data.get("sources", []),
            "methodology": finding_data.get("methodology", ""),
            "created_at": datetime.now(),
            "validation_status": "pending",  # pending, validated, disputed
            "related_findings": [],
        }

        kb.research_findings[finding_id] = finding

        # Find related findings
        self._link_related_findings(kb, finding_id)

        return True

    def search_knowledge_base(
        self, kb: KnowledgeBase, query: str, search_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """Search knowledge base content"""

        results = []
        query_lower = query.lower()

        # Search documents
        if search_type in ["all", "documents"]:
            for doc_id, document in kb.documents.items():
                if self._matches_query(document, query_lower):
                    results.append(
                        {
                            "type": "document",
                            "id": doc_id,
                            "content": document,
                            "relevance_score": self._calculate_relevance(document, query_lower),
                        }
                    )

        # Search research findings
        if search_type in ["all", "findings"]:
            for finding_id, finding in kb.research_findings.items():
                if self._matches_query(finding, query_lower):
                    results.append(
                        {
                            "type": "finding",
                            "id": finding_id,
                            "content": finding,
                            "relevance_score": self._calculate_relevance(finding, query_lower),
                        }
                    )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results

    def sync_knowledge_base(self, kb: KnowledgeBase, other_kb: KnowledgeBase) -> bool:
        """Sync knowledge base with another instance"""

        if not self.auto_sync:
            return False

        try:
            # Sync documents
            for doc_id, document in other_kb.documents.items():
                if doc_id not in kb.documents:
                    kb.documents[doc_id] = document
                elif document["last_modified"] > kb.documents[doc_id]["last_modified"]:
                    if self.conflict_resolution == "merge":
                        self._merge_documents(kb.documents[doc_id], document)
                    elif self.conflict_resolution == "overwrite":
                        kb.documents[doc_id] = document

            # Sync findings
            for finding_id, finding in other_kb.research_findings.items():
                if finding_id not in kb.research_findings:
                    kb.research_findings[finding_id] = finding
                elif finding["created_at"] > kb.research_findings[finding_id]["created_at"]:
                    if self.conflict_resolution == "overwrite":
                        kb.research_findings[finding_id] = finding

            # Update metadata
            kb.last_updated = datetime.now()
            kb.update_history.append(
                {"action": "sync_completed", "source_kb": other_kb.id, "timestamp": datetime.now()}
            )

            return True

        except Exception:
            return False

    def _auto_categorize_document(
        self, kb: KnowledgeBase, doc_id: str, document: Dict[str, Any]
    ) -> None:
        """Automatically categorize document"""

        content_text = str(document.get("content", "")).lower()
        category = document.get("category", "general")

        # Simple keyword-based categorization
        category_keywords = {
            "methodology": ["method", "approach", "technique", "procedure"],
            "analysis": ["analysis", "examine", "investigate", "study"],
            "findings": ["result", "finding", "conclusion", "discovery"],
            "sources": ["source", "reference", "citation", "bibliography"],
            "planning": ["plan", "strategy", "timeline", "milestone"],
        }

        for cat, keywords in category_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                category = cat
                break

        # Add to category
        if category not in kb.categories:
            kb.categories[category] = []
        kb.categories[category].append(doc_id)

    def _link_related_findings(self, kb: KnowledgeBase, finding_id: str) -> None:
        """Link related research findings"""

        current_finding = kb.research_findings[finding_id]
        current_data = str(current_finding.get("data", "")).lower()

        # Find related findings based on content similarity
        for other_id, other_finding in kb.research_findings.items():
            if other_id == finding_id:
                continue

            other_data = str(other_finding.get("data", "")).lower()

            # Simple similarity check
            common_words = set(current_data.split()) & set(other_data.split())
            if len(common_words) > 3:  # Threshold for relatedness
                current_finding["related_findings"].append(other_id)
                other_finding["related_findings"].append(finding_id)

    def _matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if item matches search query"""
        item_text = str(item).lower()
        return query in item_text

    def _calculate_relevance(self, item: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for search result"""
        item_text = str(item).lower()
        query_words = query.split()

        matches = sum(1 for word in query_words if word in item_text)
        return matches / len(query_words) if query_words else 0.0

    def _merge_documents(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> None:
        """Merge two document versions"""

        # Simple merge strategy - combine content and update metadata
        doc1["content"].update(doc2.get("content", {}))
        doc1["last_modified"] = max(doc1["last_modified"], doc2["last_modified"])

        # Merge tags
        doc1_tags = set(doc1.get("tags", []))
        doc2_tags = set(doc2.get("tags", []))
        doc1["tags"] = list(doc1_tags | doc2_tags)


class CollaborationOrchestrator:
    """Main orchestrator for collaborative research projects"""

    def __init__(self):
        self.task_coordinator = TaskCoordinator()
        self.peer_review = PeerReviewSystem()
        self.knowledge_manager = KnowledgeManager()
        self.projects: Dict[str, CollaborativeProject] = {}

    def create_project(
        self, name: str, description: str, collaboration_type: CollaborationType
    ) -> CollaborativeProject:
        """Create new collaborative research project"""

        project_id = f"proj_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        project = CollaborativeProject(
            id=project_id, name=name, description=description, collaboration_type=collaboration_type
        )

        self.projects[project_id] = project

        # Create knowledge base
        self.knowledge_manager.create_knowledge_base(
            project, f"{name} Knowledge Base", f"Shared knowledge for {name}"
        )

        return project

    def add_researcher(self, project_id: str, researcher: Researcher) -> bool:
        """Add researcher to project"""

        if project_id not in self.projects:
            return False

        project = self.projects[project_id]
        project.researchers[researcher.id] = researcher

        # Add to knowledge base contributors
        if project.knowledge_base:
            project.knowledge_base.contributors.add(researcher.id)

        return True

    def create_task(
        self,
        project_id: str,
        title: str,
        description: str,
        research_area: str = "",
        priority: int = 1,
    ) -> Optional[str]:
        """Create new collaborative task"""

        if project_id not in self.projects:
            return None

        task_id = f"task_{hashlib.md5(f'{project_id}_{title}'.encode()).hexdigest()[:8]}"

        task = CollaborativeTask(
            id=task_id,
            title=title,
            description=description,
            research_area=research_area,
            priority=priority,
        )

        project = self.projects[project_id]
        project.tasks[task_id] = task

        # Auto-assign if possible
        assignee = self.task_coordinator.assign_task(task, project)
        if assignee:
            task.assigned_to = assignee
            task.assigned_at = datetime.now()
            project.researchers[assignee].current_tasks.append(task_id)

        return task_id

    def complete_task(self, project_id: str, task_id: str, findings: Dict[str, Any]) -> bool:
        """Mark task as complete and initiate review if needed"""

        if project_id not in self.projects or task_id not in self.projects[project_id].tasks:
            return False

        project = self.projects[project_id]
        task = project.tasks[task_id]

        # Update task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.findings = findings

        # Move from current to completed tasks
        if task.assigned_to and task.assigned_to in project.researchers:
            researcher = project.researchers[task.assigned_to]
            if task_id in researcher.current_tasks:
                researcher.current_tasks.remove(task_id)
            researcher.completed_tasks.append(task_id)

        # Initiate peer review for high-quality tasks
        if task.quality_score >= 0.8:  # High quality threshold for review
            self.peer_review.initiate_review(task, project)

        # Add findings to knowledge base
        if project.knowledge_base and findings:
            self.knowledge_manager.add_research_finding(
                project.knowledge_base,
                f"finding_{task_id}",
                findings,
                task.assigned_to or "unknown",
            )

        return True

    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive project status"""

        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        # Calculate progress
        total_tasks = len(project.tasks)
        completed_tasks = len(
            [
                t
                for t in project.tasks.values()
                if t.status in [TaskStatus.COMPLETED, TaskStatus.APPROVED]
            ]
        )

        progress = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

        # Calculate team metrics
        team_metrics = self._calculate_team_metrics(project)

        # Task status distribution
        task_statuses = {}
        for task in project.tasks.values():
            status = task.status.value
            task_statuses[status] = task_statuses.get(status, 0) + 1

        return {
            "project_id": project_id,
            "name": project.name,
            "collaboration_type": project.collaboration_type.value,
            "progress_percentage": round(progress, 2),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "team_size": len(project.researchers),
            "task_status_distribution": task_statuses,
            "team_metrics": team_metrics,
            "knowledge_base_size": (
                len(project.knowledge_base.documents) if project.knowledge_base else 0
            ),
            "research_findings": (
                len(project.knowledge_base.research_findings) if project.knowledge_base else 0
            ),
        }

    def _calculate_team_metrics(self, project: CollaborativeProject) -> Dict[str, float]:
        """Calculate team performance metrics"""

        if not project.researchers:
            return {}

        researchers = list(project.researchers.values())

        avg_quality = sum(r.quality_score for r in researchers) / len(researchers)
        avg_reliability = sum(r.reliability_score for r in researchers) / len(researchers)
        avg_collaboration = sum(r.collaboration_score for r in researchers) / len(researchers)
        avg_communication = sum(r.communication_score for r in researchers) / len(researchers)

        return {
            "average_quality_score": round(avg_quality, 2),
            "average_reliability_score": round(avg_reliability, 2),
            "average_collaboration_score": round(avg_collaboration, 2),
            "average_communication_score": round(avg_communication, 2),
            "team_efficiency": round((avg_quality + avg_reliability + avg_collaboration) / 3, 2),
        }
