# Python Compliance Best Practices

**Objective**: Master senior-level Python compliance patterns for production systems. When you need to implement regulatory compliance, when you want to build audit-ready applications, when you need enterprise-grade compliance strategies—these best practices become your weapon of choice.

## Core Principles

- **Regulatory Compliance**: Meet industry and legal requirements
- **Data Protection**: Implement comprehensive data privacy measures
- **Audit Trail**: Maintain detailed logs and documentation
- **Risk Management**: Identify and mitigate compliance risks
- **Continuous Monitoring**: Ensure ongoing compliance

## Regulatory Compliance

### GDPR Compliance

```python
# python/01-gdpr-compliance.py

"""
GDPR compliance patterns and data protection implementation
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import uuid
from datetime import datetime, timedelta
import logging
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSubject(Enum):
    """Data subject type enumeration"""
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    VENDOR = "vendor"
    PARTNER = "partner"

class ProcessingPurpose(Enum):
    """Data processing purpose enumeration"""
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SERVICE_DELIVERY = "service_delivery"
    LEGAL_COMPLIANCE = "legal_compliance"
    RESEARCH = "research"

class DataCategory(Enum):
    """Data category enumeration"""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"

@dataclass
class PersonalData:
    """Personal data definition"""
    id: str
    data_subject_id: str
    data_category: DataCategory
    data_type: str
    value: str
    processing_purpose: ProcessingPurpose
    collected_at: datetime
    retention_period_days: int
    is_encrypted: bool = True
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    legal_basis: str = "consent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class GDPRComplianceManager:
    """GDPR compliance manager"""
    
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        self.personal_data = {}
        self.consent_records = {}
        self.data_processing_logs = []
        self.data_breach_logs = []
        self.audit_logs = []
        self.lock = threading.Lock()
    
    def collect_personal_data(self, data_subject_id: str, data_category: DataCategory,
                            data_type: str, value: str, processing_purpose: ProcessingPurpose,
                            retention_period_days: int = 2555,  # 7 years default
                            consent_given: bool = False) -> str:
        """Collect personal data with GDPR compliance"""
        with self.lock:
            # Generate unique data ID
            data_id = str(uuid.uuid4())
            
            # Encrypt sensitive data
            encrypted_value = self._encrypt_data(value) if data_category in [DataCategory.SENSITIVE, DataCategory.HEALTH, DataCategory.BIOMETRIC] else value
            
            # Create personal data record
            personal_data = PersonalData(
                id=data_id,
                data_subject_id=data_subject_id,
                data_category=data_category,
                data_type=data_type,
                value=encrypted_value,
                processing_purpose=processing_purpose,
                collected_at=datetime.utcnow(),
                retention_period_days=retention_period_days,
                is_encrypted=data_category in [DataCategory.SENSITIVE, DataCategory.HEALTH, DataCategory.BIOMETRIC],
                consent_given=consent_given,
                consent_date=datetime.utcnow() if consent_given else None
            )
            
            # Store personal data
            self.personal_data[data_id] = personal_data
            
            # Log data collection
            self._log_data_processing(
                action="collect",
                data_subject_id=data_subject_id,
                data_id=data_id,
                processing_purpose=processing_purpose.value,
                legal_basis="consent" if consent_given else "legitimate_interest"
            )
            
            logger.info(f"Personal data collected: {data_id}")
            return data_id
    
    def get_personal_data(self, data_subject_id: str) -> List[PersonalData]:
        """Get all personal data for a data subject"""
        with self.lock:
            subject_data = [
                data for data in self.personal_data.values()
                if data.data_subject_id == data_subject_id
            ]
            
            # Log data access
            self._log_data_processing(
                action="access",
                data_subject_id=data_subject_id,
                processing_purpose=ProcessingPurpose.SERVICE_DELIVERY
            )
            
            return subject_data
    
    def update_personal_data(self, data_id: str, new_value: str) -> bool:
        """Update personal data"""
        with self.lock:
            if data_id not in self.personal_data:
                return False
            
            data = self.personal_data[data_id]
            
            # Encrypt if necessary
            if data.is_encrypted:
                new_value = self._encrypt_data(new_value)
            
            # Update data
            data.value = new_value
            
            # Log data update
            self._log_data_processing(
                action="update",
                data_subject_id=data.data_subject_id,
                data_id=data_id,
                processing_purpose=data.processing_purpose
            )
            
            logger.info(f"Personal data updated: {data_id}")
            return True
    
    def delete_personal_data(self, data_id: str) -> bool:
        """Delete personal data (Right to be Forgotten)"""
        with self.lock:
            if data_id not in self.personal_data:
                return False
            
            data = self.personal_data[data_id]
            
            # Log data deletion
            self._log_data_processing(
                action="delete",
                data_subject_id=data.data_subject_id,
                data_id=data_id,
                processing_purpose=data.processing_purpose
            )
            
            # Delete data
            del self.personal_data[data_id]
            
            logger.info(f"Personal data deleted: {data_id}")
            return True
    
    def export_personal_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Export personal data (Data Portability)"""
        with self.lock:
            subject_data = self.get_personal_data(data_subject_id)
            
            # Decrypt data for export
            export_data = []
            for data in subject_data:
                export_item = data.to_dict()
                if data.is_encrypted:
                    export_item['value'] = self._decrypt_data(data.value)
                export_data.append(export_item)
            
            # Log data export
            self._log_data_processing(
                action="export",
                data_subject_id=data_subject_id,
                processing_purpose=ProcessingPurpose.SERVICE_DELIVERY
            )
            
            return {
                "data_subject_id": data_subject_id,
                "export_date": datetime.utcnow().isoformat(),
                "personal_data": export_data
            }
    
    def give_consent(self, data_subject_id: str, processing_purpose: ProcessingPurpose,
                    consent_given: bool = True) -> bool:
        """Record data subject consent"""
        with self.lock:
            consent_id = str(uuid.uuid4())
            
            consent_record = {
                "id": consent_id,
                "data_subject_id": data_subject_id,
                "processing_purpose": processing_purpose.value,
                "consent_given": consent_given,
                "consent_date": datetime.utcnow(),
                "ip_address": "unknown",  # Would be captured from request
                "user_agent": "unknown"   # Would be captured from request
            }
            
            self.consent_records[consent_id] = consent_record
            
            # Update related personal data
            for data in self.personal_data.values():
                if (data.data_subject_id == data_subject_id and 
                    data.processing_purpose == processing_purpose):
                    data.consent_given = consent_given
                    data.consent_date = datetime.utcnow()
            
            # Log consent
            self._log_data_processing(
                action="consent",
                data_subject_id=data_subject_id,
                processing_purpose=processing_purpose
            )
            
            logger.info(f"Consent recorded: {consent_id}")
            return True
    
    def withdraw_consent(self, data_subject_id: str, processing_purpose: ProcessingPurpose) -> bool:
        """Withdraw data subject consent"""
        return self.give_consent(data_subject_id, processing_purpose, False)
    
    def check_retention_periods(self) -> List[Dict[str, Any]]:
        """Check for data that should be deleted based on retention periods"""
        with self.lock:
            expired_data = []
            current_time = datetime.utcnow()
            
            for data_id, data in self.personal_data.items():
                retention_end = data.collected_at + timedelta(days=data.retention_period_days)
                
                if current_time > retention_end:
                    expired_data.append({
                        "data_id": data_id,
                        "data_subject_id": data.data_subject_id,
                        "data_type": data.data_type,
                        "collected_at": data.collected_at.isoformat(),
                        "retention_end": retention_end.isoformat(),
                        "days_overdue": (current_time - retention_end).days
                    })
            
            return expired_data
    
    def auto_delete_expired_data(self) -> int:
        """Automatically delete expired data"""
        expired_data = self.check_retention_periods()
        deleted_count = 0
        
        for expired in expired_data:
            if self.delete_personal_data(expired["data_id"]):
                deleted_count += 1
                logger.info(f"Auto-deleted expired data: {expired['data_id']}")
        
        return deleted_count
    
    def report_data_breach(self, breach_description: str, affected_data_subjects: List[str],
                          breach_date: datetime, discovered_date: datetime) -> str:
        """Report data breach"""
        breach_id = str(uuid.uuid4())
        
        breach_record = {
            "id": breach_id,
            "description": breach_description,
            "affected_data_subjects": affected_data_subjects,
            "breach_date": breach_date,
            "discovered_date": discovered_date,
            "reported_date": datetime.utcnow(),
            "status": "reported"
        }
        
        self.data_breach_logs.append(breach_record)
        
        # Log breach
        self._log_data_processing(
            action="breach_report",
            data_subject_id="system",
            processing_purpose=ProcessingPurpose.LEGAL_COMPLIANCE
        )
        
        logger.warning(f"Data breach reported: {breach_id}")
        return breach_id
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        # Simple encryption for demonstration
        # In production, use proper encryption libraries
        return hashlib.sha256((data + self.encryption_key).encode()).hexdigest()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data (simplified for demonstration)"""
        # In production, implement proper decryption
        return f"decrypted_{encrypted_data}"
    
    def _log_data_processing(self, action: str, data_subject_id: str, 
                           data_id: Optional[str] = None, processing_purpose: Optional[ProcessingPurpose] = None,
                           legal_basis: Optional[str] = None) -> None:
        """Log data processing activity"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "action": action,
            "data_subject_id": data_subject_id,
            "data_id": data_id,
            "processing_purpose": processing_purpose.value if processing_purpose else None,
            "legal_basis": legal_basis,
            "ip_address": "unknown",
            "user_agent": "unknown"
        }
        
        self.data_processing_logs.append(log_entry)
        self.audit_logs.append(log_entry)
    
    def get_audit_trail(self, data_subject_id: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit trail for compliance"""
        logs = self.audit_logs.copy()
        
        if data_subject_id:
            logs = [log for log in logs if log["data_subject_id"] == data_subject_id]
        
        if start_date:
            logs = [log for log in logs if log["timestamp"] >= start_date]
        
        if end_date:
            logs = [log for log in logs if log["timestamp"] <= end_date]
        
        return logs

class HIPAAComplianceManager:
    """HIPAA compliance manager for healthcare data"""
    
    def __init__(self):
        self.phi_data = {}  # Protected Health Information
        self.access_logs = []
        self.breach_logs = []
        self.audit_logs = []
        self.lock = threading.Lock()
    
    def store_phi(self, patient_id: str, phi_type: str, value: str,
                  access_level: str = "restricted") -> str:
        """Store Protected Health Information"""
        with self.lock:
            phi_id = str(uuid.uuid4())
            
            phi_record = {
                "id": phi_id,
                "patient_id": patient_id,
                "phi_type": phi_type,
                "value": value,
                "access_level": access_level,
                "created_at": datetime.utcnow(),
                "is_encrypted": True
            }
            
            self.phi_data[phi_id] = phi_record
            
            # Log PHI access
            self._log_phi_access("store", patient_id, phi_id, phi_type)
            
            logger.info(f"PHI stored: {phi_id}")
            return phi_id
    
    def access_phi(self, patient_id: str, phi_id: str, user_id: str, 
                   access_reason: str) -> Optional[str]:
        """Access Protected Health Information"""
        with self.lock:
            if phi_id not in self.phi_data:
                return None
            
            phi_record = self.phi_data[phi_id]
            
            # Log PHI access
            self._log_phi_access("access", patient_id, phi_id, phi_record["phi_type"], 
                               user_id, access_reason)
            
            return phi_record["value"]
    
    def _log_phi_access(self, action: str, patient_id: str, phi_id: str, 
                       phi_type: str, user_id: Optional[str] = None,
                       access_reason: Optional[str] = None) -> None:
        """Log PHI access for HIPAA compliance"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "action": action,
            "patient_id": patient_id,
            "phi_id": phi_id,
            "phi_type": phi_type,
            "user_id": user_id,
            "access_reason": access_reason,
            "ip_address": "unknown",
            "user_agent": "unknown"
        }
        
        self.access_logs.append(log_entry)
        self.audit_logs.append(log_entry)

class SOXComplianceManager:
    """SOX compliance manager for financial data"""
    
    def __init__(self):
        self.financial_data = {}
        self.control_logs = []
        self.audit_logs = []
        self.lock = threading.Lock()
    
    def record_financial_transaction(self, transaction_id: str, amount: float,
                                   account_id: str, transaction_type: str,
                                   user_id: str) -> bool:
        """Record financial transaction with SOX controls"""
        with self.lock:
            transaction_record = {
                "id": transaction_id,
                "amount": amount,
                "account_id": account_id,
                "transaction_type": transaction_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow(),
                "is_approved": False,
                "approval_required": amount > 10000  # SOX threshold
            }
            
            self.financial_data[transaction_id] = transaction_record
            
            # Log transaction
            self._log_financial_activity("transaction", transaction_id, user_id)
            
            logger.info(f"Financial transaction recorded: {transaction_id}")
            return True
    
    def approve_transaction(self, transaction_id: str, approver_id: str) -> bool:
        """Approve financial transaction"""
        with self.lock:
            if transaction_id not in self.financial_data:
                return False
            
            transaction = self.financial_data[transaction_id]
            transaction["is_approved"] = True
            transaction["approver_id"] = approver_id
            transaction["approval_date"] = datetime.utcnow()
            
            # Log approval
            self._log_financial_activity("approval", transaction_id, approver_id)
            
            logger.info(f"Transaction approved: {transaction_id}")
            return True
    
    def _log_financial_activity(self, activity_type: str, transaction_id: str, user_id: str) -> None:
        """Log financial activity for SOX compliance"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "activity_type": activity_type,
            "transaction_id": transaction_id,
            "user_id": user_id,
            "ip_address": "unknown",
            "user_agent": "unknown"
        }
        
        self.control_logs.append(log_entry)
        self.audit_logs.append(log_entry)

# Usage examples
def example_gdpr_compliance():
    """Example GDPR compliance usage"""
    # Create GDPR compliance manager
    gdpr_manager = GDPRComplianceManager("encryption-key")
    
    # Collect personal data
    data_id = gdpr_manager.collect_personal_data(
        data_subject_id="user123",
        data_category=DataCategory.PERSONAL,
        data_type="email",
        value="user@example.com",
        processing_purpose=ProcessingPurpose.SERVICE_DELIVERY,
        consent_given=True
    )
    print(f"Personal data collected: {data_id}")
    
    # Give consent
    gdpr_manager.give_consent("user123", ProcessingPurpose.MARKETING, True)
    
    # Get personal data
    personal_data = gdpr_manager.get_personal_data("user123")
    print(f"Personal data count: {len(personal_data)}")
    
    # Export personal data
    export_data = gdpr_manager.export_personal_data("user123")
    print(f"Export data keys: {list(export_data.keys())}")
    
    # Check retention periods
    expired_data = gdpr_manager.check_retention_periods()
    print(f"Expired data count: {len(expired_data)}")
    
    # Report data breach
    breach_id = gdpr_manager.report_data_breach(
        "Unauthorized access to user database",
        ["user123", "user456"],
        datetime.utcnow() - timedelta(days=1),
        datetime.utcnow()
    )
    print(f"Data breach reported: {breach_id}")
    
    # Get audit trail
    audit_trail = gdpr_manager.get_audit_trail("user123")
    print(f"Audit trail entries: {len(audit_trail)}")

def example_hipaa_compliance():
    """Example HIPAA compliance usage"""
    # Create HIPAA compliance manager
    hipaa_manager = HIPAAComplianceManager()
    
    # Store PHI
    phi_id = hipaa_manager.store_phi(
        patient_id="patient123",
        phi_type="medical_record",
        value="Patient has diabetes",
        access_level="restricted"
    )
    print(f"PHI stored: {phi_id}")
    
    # Access PHI
    phi_value = hipaa_manager.access_phi(
        patient_id="patient123",
        phi_id=phi_id,
        user_id="doctor456",
        access_reason="treatment"
    )
    print(f"PHI accessed: {phi_value}")

def example_sox_compliance():
    """Example SOX compliance usage"""
    # Create SOX compliance manager
    sox_manager = SOXComplianceManager()
    
    # Record financial transaction
    transaction_id = sox_manager.record_financial_transaction(
        transaction_id="txn123",
        amount=15000.00,
        account_id="acc456",
        transaction_type="payment",
        user_id="user789"
    )
    print(f"Transaction recorded: {transaction_id}")
    
    # Approve transaction
    approval_success = sox_manager.approve_transaction("txn123", "approver123")
    print(f"Transaction approved: {approval_success}")
```

### Audit and Monitoring

```python
# python/02-audit-monitoring.py

"""
Audit and monitoring patterns for compliance
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import json
import time
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AuditLevel(Enum):
    """Audit level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    """Compliance framework enumeration"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

@dataclass
class AuditEvent:
    """Audit event definition"""
    id: str
    timestamp: datetime
    event_type: str
    user_id: str
    resource_id: str
    action: str
    result: str
    details: Dict[str, Any]
    compliance_framework: ComplianceFramework
    audit_level: AuditLevel
    ip_address: str
    user_agent: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ComplianceAuditor:
    """Compliance auditor for audit trail management"""
    
    def __init__(self):
        self.audit_events = []
        self.compliance_rules = {}
        self.alert_rules = {}
        self.audit_lock = threading.Lock()
        self.initialize_compliance_rules()
    
    def initialize_compliance_rules(self) -> None:
        """Initialize compliance rules"""
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                "data_retention_days": 2555,  # 7 years
                "consent_required": True,
                "data_encryption_required": True,
                "breach_notification_hours": 72
            },
            ComplianceFramework.HIPAA: {
                "phi_encryption_required": True,
                "access_logging_required": True,
                "breach_notification_days": 60
            },
            ComplianceFramework.SOX: {
                "financial_controls_required": True,
                "approval_threshold": 10000,
                "audit_trail_retention_years": 7
            }
        }
    
    def log_audit_event(self, event_type: str, user_id: str, resource_id: str,
                       action: str, result: str, details: Dict[str, Any],
                       compliance_framework: ComplianceFramework,
                       audit_level: AuditLevel = AuditLevel.MEDIUM,
                       ip_address: str = "unknown",
                       user_agent: str = "unknown") -> str:
        """Log audit event"""
        with self.audit_lock:
            event_id = f"audit_{int(time.time())}_{user_id}"
            
            audit_event = AuditEvent(
                id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                resource_id=resource_id,
                action=action,
                result=result,
                details=details,
                compliance_framework=compliance_framework,
                audit_level=audit_level,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.audit_events.append(audit_event)
            
            # Check for compliance violations
            self._check_compliance_violations(audit_event)
            
            logger.info(f"Audit event logged: {event_id}")
            return event_id
    
    def _check_compliance_violations(self, audit_event: AuditEvent) -> None:
        """Check for compliance violations"""
        framework = audit_event.compliance_framework
        rules = self.compliance_rules.get(framework, {})
        
        # Check for high-risk activities
        if audit_event.audit_level == AuditLevel.CRITICAL:
            self._trigger_compliance_alert(audit_event, "Critical audit event detected")
        
        # Check for unauthorized access
        if audit_event.result == "unauthorized":
            self._trigger_compliance_alert(audit_event, "Unauthorized access attempt")
        
        # Check for data breaches
        if "breach" in audit_event.action.lower():
            self._trigger_compliance_alert(audit_event, "Potential data breach detected")
    
    def _trigger_compliance_alert(self, audit_event: AuditEvent, message: str) -> None:
        """Trigger compliance alert"""
        alert = {
            "timestamp": datetime.utcnow(),
            "audit_event_id": audit_event.id,
            "message": message,
            "compliance_framework": audit_event.compliance_framework.value,
            "audit_level": audit_event.audit_level.value,
            "user_id": audit_event.user_id,
            "resource_id": audit_event.resource_id
        }
        
        logger.warning(f"Compliance alert: {message}")
        # In production, this would send notifications to compliance team
    
    def get_audit_trail(self, user_id: Optional[str] = None,
                       compliance_framework: Optional[ComplianceFramework] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       audit_level: Optional[AuditLevel] = None) -> List[Dict[str, Any]]:
        """Get audit trail with filtering"""
        with self.audit_lock:
            events = [event.to_dict() for event in self.audit_events]
        
        # Apply filters
        if user_id:
            events = [event for event in events if event["user_id"] == user_id]
        
        if compliance_framework:
            events = [event for event in events if event["compliance_framework"] == compliance_framework]
        
        if start_date:
            events = [event for event in events if event["timestamp"] >= start_date]
        
        if end_date:
            events = [event for event in events if event["timestamp"] <= end_date]
        
        if audit_level:
            events = [event for event in events if event["audit_level"] == audit_level]
        
        return events
    
    def generate_compliance_report(self, compliance_framework: ComplianceFramework,
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        events = self.get_audit_trail(
            compliance_framework=compliance_framework,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate compliance metrics
        total_events = len(events)
        critical_events = len([e for e in events if e["audit_level"] == AuditLevel.CRITICAL.value])
        unauthorized_events = len([e for e in events if e["result"] == "unauthorized"])
        
        # Group by event type
        event_types = {}
        for event in events:
            event_type = event["event_type"]
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Group by user
        user_activities = {}
        for event in events:
            user_id = event["user_id"]
            if user_id not in user_activities:
                user_activities[user_id] = 0
            user_activities[user_id] += 1
        
        return {
            "compliance_framework": compliance_framework.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "critical_events": critical_events,
                "unauthorized_events": unauthorized_events,
                "compliance_score": self._calculate_compliance_score(events)
            },
            "event_types": event_types,
            "user_activities": user_activities,
            "recommendations": self._generate_recommendations(events)
        }
    
    def _calculate_compliance_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate compliance score"""
        if not events:
            return 100.0
        
        total_events = len(events)
        critical_events = len([e for e in events if e["audit_level"] == AuditLevel.CRITICAL.value])
        unauthorized_events = len([e for e in events if e["result"] == "unauthorized"])
        
        # Calculate score (higher is better)
        score = 100.0
        score -= (critical_events / total_events) * 50  # Critical events heavily penalized
        score -= (unauthorized_events / total_events) * 30  # Unauthorized events penalized
        
        return max(0.0, score)
    
    def _generate_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check for high unauthorized access
        unauthorized_events = [e for e in events if e["result"] == "unauthorized"]
        if len(unauthorized_events) > 10:
            recommendations.append("High number of unauthorized access attempts. Review access controls.")
        
        # Check for critical events
        critical_events = [e for e in events if e["audit_level"] == AuditLevel.CRITICAL.value]
        if len(critical_events) > 5:
            recommendations.append("Multiple critical events detected. Review security measures.")
        
        # Check for data access patterns
        data_access_events = [e for e in events if "data" in e["action"].lower()]
        if len(data_access_events) > 100:
            recommendations.append("High data access volume. Consider implementing data access controls.")
        
        return recommendations

class ComplianceMonitor:
    """Compliance monitoring and alerting"""
    
    def __init__(self, auditor: ComplianceAuditor):
        self.auditor = auditor
        self.monitoring_rules = {}
        self.alert_handlers = []
        self.is_monitoring = False
        self.monitoring_thread = None
    
    def add_monitoring_rule(self, rule_name: str, condition: callable, 
                           alert_message: str, compliance_framework: ComplianceFramework) -> None:
        """Add monitoring rule"""
        self.monitoring_rules[rule_name] = {
            "condition": condition,
            "alert_message": alert_message,
            "compliance_framework": compliance_framework
        }
        logger.info(f"Monitoring rule added: {rule_name}")
    
    def start_monitoring(self) -> None:
        """Start compliance monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Compliance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop compliance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Compliance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_monitoring_rules()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_monitoring_rules(self) -> None:
        """Check all monitoring rules"""
        for rule_name, rule in self.monitoring_rules.items():
            try:
                if rule["condition"]():
                    self._trigger_alert(rule_name, rule["alert_message"], rule["compliance_framework"])
            except Exception as e:
                logger.error(f"Error checking monitoring rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule_name: str, message: str, compliance_framework: ComplianceFramework) -> None:
        """Trigger compliance alert"""
        alert = {
            "timestamp": datetime.utcnow(),
            "rule_name": rule_name,
            "message": message,
            "compliance_framework": compliance_framework.value,
            "severity": "high"
        }
        
        logger.warning(f"Compliance alert triggered: {message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: callable) -> None:
        """Add alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Alert handler added")

# Usage examples
def example_compliance_auditing():
    """Example compliance auditing usage"""
    # Create compliance auditor
    auditor = ComplianceAuditor()
    
    # Log audit events
    auditor.log_audit_event(
        event_type="data_access",
        user_id="user123",
        resource_id="data456",
        action="read",
        result="success",
        details={"data_type": "personal", "sensitive": True},
        compliance_framework=ComplianceFramework.GDPR,
        audit_level=AuditLevel.HIGH
    )
    
    # Get audit trail
    audit_trail = auditor.get_audit_trail(
        compliance_framework=ComplianceFramework.GDPR,
        start_date=datetime.utcnow() - timedelta(days=7)
    )
    print(f"Audit trail entries: {len(audit_trail)}")
    
    # Generate compliance report
    report = auditor.generate_compliance_report(
        ComplianceFramework.GDPR,
        datetime.utcnow() - timedelta(days=30),
        datetime.utcnow()
    )
    print(f"Compliance score: {report['summary']['compliance_score']}")
    
    # Create compliance monitor
    monitor = ComplianceMonitor(auditor)
    
    # Add monitoring rule
    def check_unauthorized_access():
        events = auditor.get_audit_trail(
            start_date=datetime.utcnow() - timedelta(hours=1)
        )
        unauthorized = [e for e in events if e["result"] == "unauthorized"]
        return len(unauthorized) > 5
    
    monitor.add_monitoring_rule(
        "unauthorized_access",
        check_unauthorized_access,
        "High number of unauthorized access attempts detected",
        ComplianceFramework.GDPR
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Stop monitoring
    monitor.stop_monitoring()
```

## TL;DR Runbook

### Quick Start

```python
# 1. GDPR compliance
gdpr_manager = GDPRComplianceManager("encryption-key")
data_id = gdpr_manager.collect_personal_data("user123", DataCategory.PERSONAL, "email", "user@example.com", ProcessingPurpose.SERVICE_DELIVERY)

# 2. HIPAA compliance
hipaa_manager = HIPAAComplianceManager()
phi_id = hipaa_manager.store_phi("patient123", "medical_record", "Patient has diabetes")

# 3. SOX compliance
sox_manager = SOXComplianceManager()
sox_manager.record_financial_transaction("txn123", 15000.00, "acc456", "payment", "user789")

# 4. Compliance auditing
auditor = ComplianceAuditor()
auditor.log_audit_event("data_access", "user123", "data456", "read", "success", {}, ComplianceFramework.GDPR)

# 5. Compliance monitoring
monitor = ComplianceMonitor(auditor)
monitor.add_monitoring_rule("unauthorized_access", check_unauthorized_access, "High unauthorized access", ComplianceFramework.GDPR)
monitor.start_monitoring()
```

### Essential Patterns

```python
# Complete compliance setup
def setup_compliance():
    """Setup complete compliance environment"""
    
    # GDPR compliance
    gdpr_manager = GDPRComplianceManager("encryption-key")
    
    # HIPAA compliance
    hipaa_manager = HIPAAComplianceManager()
    
    # SOX compliance
    sox_manager = SOXComplianceManager()
    
    # Compliance auditing
    auditor = ComplianceAuditor()
    
    # Compliance monitoring
    monitor = ComplianceMonitor(auditor)
    
    print("Compliance setup complete!")
```

---

*This guide provides the complete machinery for Python compliance best practices. Each pattern includes implementation examples, compliance strategies, and real-world usage patterns for enterprise compliance management.*
