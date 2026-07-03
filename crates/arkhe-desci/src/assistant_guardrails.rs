//! Guardrails para o assistente de IA do DeSciOS

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

use crate::error::{DesciError, Result};

/// Erro específico dos guardrails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardrailError {
    InvariantViolation { invariant_id: String, detail: String },
    PiiDetected { pii_types: Vec<String> },
    ContentBlocked { category: String, reason: String },
    LlmUnavailable { detail: String },
    Timeout { seconds: u64 },
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvariantViolation { invariant_id, detail } =>
                write!(f, "Invariant {} violated: {}", invariant_id, detail),
            Self::PiiDetected { pii_types } =>
                write!(f, "PII detected: {}", pii_types.join(", ")),
            Self::ContentBlocked { category, reason } =>
                write!(f, "Content blocked [{}]: {}", category, reason),
            Self::LlmUnavailable { detail } =>
                write!(f, "LLM unavailable: {}", detail),
            Self::Timeout { seconds } =>
                write!(f, "Guardrail timeout after {}s", seconds),
        }
    }
}
impl std::error::Error for GuardrailError {}

/// Contexto de uma interação do assistente
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantContext {
    pub user_id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub active_tools: Vec<String>,
    pub workspace_path: String,
}

impl Default for AssistantContext {
    fn default() -> Self {
        Self {
            user_id: "anonymous".to_string(),
            session_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            active_tools: Vec::new(),
            workspace_path: "/home/deScier".to_string(),
        }
    }
}

/// Tipo de PII detectado
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PiiType {
    Email,
    PhoneNumber,
    Cpf,        // CPF brasileiro
    CreditCard,
    Ssn,        // SSN americano
    IpAddress,
    Passport,
    BankAccount,
    Custom(String),
}

impl std::fmt::Display for PiiType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Email => write!(f, "email"),
            Self::PhoneNumber => write!(f, "phone_number"),
            Self::Cpf => write!(f, "cpf"),
            Self::CreditCard => write!(f, "credit_card"),
            Self::Ssn => write!(f, "ssn"),
            Self::IpAddress => write!(f, "ip_address"),
            Self::Passport => write!(f, "passport"),
            Self::BankAccount => write!(f, "bank_account"),
            Self::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

/// Uma redação de PII
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Redaction {
    pub pii_type: PiiType,
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub masked: String,
}

/// Resultado da verificação de PII
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiCheckResult {
    pub masked_text: String,
    pub redactions: Vec<Redaction>,
    pub has_pii: bool,
}

/// Masker de PII baseado em regex
#[derive(Debug, Clone)]
pub struct PiiMasker {
    patterns: Vec<(PiiType, regex::Regex, String)>,
}

impl PiiMasker {
    pub fn new() -> Self {
        let patterns = vec![
            (PiiType::Email,
             regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
             "[EMAIL_REDACTED]".to_string()),
            (PiiType::Cpf,
             regex::Regex::new(r"\b\d{3}[.]?\d{3}[.]?\d{3}[-]?\d{2}\b").unwrap(),
             "[CPF_REDACTED]".to_string()),
            (PiiType::PhoneNumber,
             regex::Regex::new(r"\(?\d{2}\)?\s?\d{4,5}[-.]?\d{4}").unwrap(),
             "[PHONE_REDACTED]".to_string()),
            (PiiType::CreditCard,
             regex::Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
             "[CC_REDACTED]".to_string()),
            (PiiType::IpAddress,
             regex::Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
             "[IP_REDACTED]".to_string()),
            (PiiType::Ssn,
             regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
             "[SSN_REDACTED]".to_string()),
        ];
        Self { patterns }
    }

    pub fn mask(&self, text: &str) -> PiiCheckResult {
        let mut masked = text.to_string();
        let mut redactions = Vec::new();

        let mut all_matches: Vec<(usize, usize, &PiiType, &str)> = Vec::new();
        for (pii_type, regex, replacement) in &self.patterns {
            for mat in regex.find_iter(text) {
                all_matches.push((mat.start(), mat.end(), pii_type, replacement.as_str()));
            }
        }

        all_matches.sort_by(|a, b| b.0.cmp(&a.0));

        for (start, end, pii_type, replacement) in all_matches {
            let original = text[start..end].to_string();
            masked = format!("{}{}{}", &masked[..start], replacement, &masked[end..]);
            redactions.push(Redaction {
                pii_type: pii_type.clone(),
                start,
                end,
                original,
                masked: replacement.to_string(),
            });
        }

        redactions.reverse();

        let has_pii = !redactions.is_empty();

        PiiCheckResult {
            masked_text: masked,
            redactions,
            has_pii,
        }
    }
}

impl Default for PiiMasker {
    fn default() -> Self {
        Self::new()
    }
}

/// Categorias de conteúdo para guardrails
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GuardrailCategory {
    HarmfulContent,
    PiiExfiltration,
    SystemExploitation,
    DataDestruction,
    UnauthorizedAccess,
    Custom(String),
}

impl std::fmt::Display for GuardrailCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HarmfulContent => write!(f, "harmful_content"),
            Self::PiiExfiltration => write!(f, "pii_exfiltration"),
            Self::SystemExploitation => write!(f, "system_exploitation"),
            Self::DataDestruction => write!(f, "data_destruction"),
            Self::UnauthorizedAccess => write!(f, "unauthorized_access"),
            Self::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailCheckResult {
    pub safe: bool,
    pub category: Option<GuardrailCategory>,
    pub reason: Option<String>,
    pub risk_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailConfig {
    pub pii_masking_enabled: bool,
    pub content_check_enabled: bool,
    pub risk_threshold: f32,
    pub blocked_patterns: Vec<String>,
    pub allowed_domains: Vec<String>,
    pub timeout_seconds: u64,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            pii_masking_enabled: true,
            content_check_enabled: true,
            risk_threshold: 0.7,
            blocked_patterns: vec![
                r"rm\s+-rf\s+/".to_string(),
                r"mkfs\.".to_string(),
                r"dd\s+if=/dev/zero".to_string(),
                r">\s*/dev/sd".to_string(),
                r"chmod\s+777\s+/".to_string(),
                r"curl.*\|\s*(ba)?sh".to_string(),
            ],
            allowed_domains: vec![
                "ncbi.nlm.nih.gov".to_string(),
                "ensembl.org".to_string(),
                "uniprot.org".to_string(),
                "github.com".to_string(),
            ],
            timeout_seconds: 10,
        }
    }
}

pub struct DeSciAssistantGuardrails {
    config: GuardrailConfig,
    pii_masker: PiiMasker,
    blocked_regexes: Vec<regex::Regex>,
}

impl DeSciAssistantGuardrails {
    pub fn new() -> Self {
        Self::with_config(GuardrailConfig::default())
    }

    pub fn with_config(config: GuardrailConfig) -> Self {
        let blocked_regexes: Vec<regex::Regex> = config
            .blocked_patterns
            .iter()
            .filter_map(|p| regex::Regex::new(p).ok())
            .collect();

        Self {
            config,
            pii_masker: PiiMasker::new(),
            blocked_regexes,
        }
    }

    pub fn check_message(
        &self,
        message: &str,
        _context: &AssistantContext,
    ) -> Result<(String, GuardrailCheckResult)> {
        for regex in &self.blocked_regexes {
            if regex.is_match(message) {
                warn!(pattern = %regex.as_str(), "Blocked pattern detected");
                return Ok((
                    "[CONTENT_BLOCKED]".to_string(),
                    GuardrailCheckResult {
                        safe: false,
                        category: Some(GuardrailCategory::SystemExploitation),
                        reason: Some("Message matches a blocked pattern".to_string()),
                        risk_score: 1.0,
                    },
                ));
            }
        }

        let processed = if self.config.pii_masking_enabled {
            let result = self.pii_masker.mask(message);
            if result.has_pii {
                info!(
                    pii_count = result.redactions.len(),
                    "PII detected and masked"
                );
            }
            result.masked_text
        } else {
            message.to_string()
        };

        let risk_score = self.compute_local_risk(&processed);

        let check_result = if risk_score >= self.config.risk_threshold {
            GuardrailCheckResult {
                safe: false,
                category: Some(GuardrailCategory::HarmfulContent),
                reason: Some(format!("Risk score {:.2} exceeds threshold {:.2}", risk_score, self.config.risk_threshold)),
                risk_score,
            }
        } else {
            GuardrailCheckResult {
                safe: true,
                category: None,
                reason: None,
                risk_score,
            }
        };

        Ok((processed, check_result))
    }

    pub fn check_url(&self, url: &str) -> Result<GuardrailCheckResult> {
        let parsed = url::Url::parse(url)
            .map_err(|e| DesciError::Parse(format!("Invalid URL: {}", e)))?;

        let host = parsed.host_str().unwrap_or("");
        let blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"];

        if blocked_hosts.contains(&host) {
            return Ok(GuardrailCheckResult {
                safe: false,
                category: Some(GuardrailCategory::UnauthorizedAccess),
                reason: Some("Internal URL blocked (SSRF prevention)".to_string()),
                risk_score: 1.0,
            });
        }

        if let Ok(addr) = std::str::FromStr::from_str(host) {
            if is_private_ip(&addr) {
                return Ok(GuardrailCheckResult {
                    safe: false,
                    category: Some(GuardrailCategory::UnauthorizedAccess),
                    reason: Some("Private IP address blocked".to_string()),
                    risk_score: 1.0,
                });
            }
        }

        Ok(GuardrailCheckResult {
            safe: true,
            category: None,
            reason: None,
            risk_score: 0.0,
        })
    }

    pub fn config(&self) -> &GuardrailConfig {
        &self.config
    }

    fn compute_local_risk(&self, text: &str) -> f32 {
        let mut score: f32 = 0.0;
        let lower = text.to_lowercase();

        let risk_indicators: &[(&str, f32)] = &[
            ("delete all", 0.8),
            ("drop table", 0.9),
            ("format disk", 0.9),
            ("overwrite", 0.4),
            ("bypass", 0.5),
            ("ignore warning", 0.3),
            ("sudo", 0.3),
            ("root", 0.2),
            ("password", 0.2),
            ("secret", 0.3),
            ("api key", 0.4),
            ("token", 0.2),
            ("credential", 0.4),
        ];

        for (indicator, weight) in risk_indicators {
            if lower.contains(indicator) {
                score = score.max(*weight);
            }
        }

        let sci_context: &[&str] = &[
            "gene", "protein", "sequence", "alignment", "blast",
            "genome", "transcript", "expression", "pathway",
            "jupyter", "notebook", "analysis", "dataset",
        ];

        let sci_count = sci_context.iter().filter(|s| lower.contains(*s)).count();
        if sci_count > 0 {
            score *= 0.5;
        }

        score.min(1.0)
    }
}

impl Default for DeSciAssistantGuardrails {
    fn default() -> Self {
        Self::new()
    }
}

fn is_private_ip(addr: &std::net::IpAddr) -> bool {
    match addr {
        std::net::IpAddr::V4(v4) => {
            v4.is_private() || v4.is_loopback() || v4.is_link_local()
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
        }
    }
}

fn from_str(s: &str) -> std::result::Result<std::net::IpAddr, ()> {
    s.parse().map_err(|_| ())
}
