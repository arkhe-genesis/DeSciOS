//! Plugin Governance — validação

use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use crate::error::DesciError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub source: String,
    pub signature: Option<String>,
    pub install_script: String,
    pub requested_permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub plugin_id: String,
    pub passed: bool,
    pub checks: Vec<ValidationCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub invariant_id: String,
    pub passed: bool,
    pub message: String,
}

pub struct PluginValidator {
    required_signatures: bool,
    max_permissions: usize,
}

impl Default for PluginValidator {
    fn default() -> Self {
        Self {
            required_signatures: false,
            max_permissions: 3,
        }
    }
}

impl PluginValidator {
    pub fn validate(&self, manifest: &PluginManifest) -> Result<ValidationResult, DesciError> {
        let mut checks = Vec::new();
        let mut passed = true;

        // OWASP-003: Provenance
        if self.required_signatures && manifest.signature.is_none() {
            passed = false;
            checks.push(ValidationCheck {
                invariant_id: "OWASP-003".to_string(),
                passed: false,
                message: "Plugin not signed".to_string(),
            });
        } else {
            checks.push(ValidationCheck {
                invariant_id: "OWASP-003".to_string(),
                passed: true,
                message: "Signature OK".to_string(),
            });
        }

        // CNT-002: Workspace confinement
        let dangerous = ["/etc/passwd", "/root/", "sudo", "rm -rf"];
        if dangerous.iter().any(|p| manifest.install_script.contains(p)) {
            passed = false;
            checks.push(ValidationCheck {
                invariant_id: "CNT-002".to_string(),
                passed: false,
                message: "Dangerous command in install_script".to_string(),
            });
        } else {
            checks.push(ValidationCheck {
                invariant_id: "CNT-002".to_string(),
                passed: true,
                message: "No dangerous commands".to_string(),
            });
        }

        // OWASP-006: Least privilege
        if manifest.requested_permissions.len() > self.max_permissions {
            passed = false;
            checks.push(ValidationCheck {
                invariant_id: "OWASP-006".to_string(),
                passed: false,
                message: format!("Too many permissions: {}", manifest.requested_permissions.len()),
            });
        } else {
            checks.push(ValidationCheck {
                invariant_id: "OWASP-006".to_string(),
                passed: true,
                message: "Permissions OK".to_string(),
            });
        }

        let summary = if passed { "Valid" } else { "Failed" };
        if !passed { warn!(plugin = %manifest.name, "Plugin validation failed"); }

        Ok(ValidationResult { plugin_id: manifest.id.clone(), passed, checks })
    }
}
