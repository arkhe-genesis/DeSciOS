//! ARKHE × DeSciOS — Integração para ciência descentralizada

pub mod error;
pub mod plugin_governance;
pub mod assistant_guardrails;
pub mod workflow_traceability;
pub mod publishing;

pub use error::{DesciError, Result};
pub use plugin_governance::{PluginValidator, PluginManifest, ValidationResult};
pub use assistant_guardrails::{DeSciAssistantGuardrails, AssistantContext, GuardrailError};
pub use workflow_traceability::{ScientificWorkflowTrace, WorkflowStep};
pub use publishing::{DeSciPublisher, DatasetMetadata};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
