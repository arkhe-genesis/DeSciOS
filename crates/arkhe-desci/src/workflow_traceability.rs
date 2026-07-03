//! Rastreabilidade IC16 para workflows científicos DeSciOS

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::info;

use crate::error::{DesciError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StepId(String);

impl StepId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StepId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed { error: String },
    Skipped { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub id: StepId,
    pub name: String,
    pub tool: String,
    pub status: StepStatus,
    pub parameters: serde_json::Value,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub hash: Option<String>,
}

impl WorkflowStep {
    pub fn new(id: impl Into<String>, name: &str, tool: &str) -> Self {
        Self {
            id: StepId::new(id),
            name: name.to_string(),
            tool: tool.to_string(),
            status: StepStatus::Pending,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
            inputs: Vec::new(),
            outputs: Vec::new(),
            started_at: None,
            completed_at: None,
            hash: None,
        }
    }

    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }

    pub fn with_inputs(mut self, inputs: Vec<String>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn start(&mut self) {
        self.status = StepStatus::Running;
        self.started_at = Some(Utc::now());
    }

    pub fn complete(&mut self, outputs: Vec<String>) {
        self.status = StepStatus::Completed;
        self.outputs = outputs;
        self.completed_at = Some(Utc::now());
    }

    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = StepStatus::Failed { error: error.into() };
        self.completed_at = Some(Utc::now());
    }

    pub fn compute_hash(&self) -> String {
        let mut map = BTreeMap::new();
        map.insert("id", serde_json::Value::String(self.id.0.clone()));
        map.insert("name", serde_json::Value::String(self.name.clone()));
        map.insert("tool", serde_json::Value::String(self.tool.clone()));
        map.insert("parameters", self.parameters.clone());
        map.insert("inputs", serde_json::to_value(&self.inputs).unwrap_or_default());
        map.insert("outputs", serde_json::to_value(&self.outputs).unwrap_or_default());

        let canonical = serde_json::to_string(&map).unwrap_or_default();
        let hash = blake3::hash(canonical.as_bytes());
        hash.to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificWorkflowTrace {
    pub trace_id: String,
    pub workflow_name: String,
    pub workflow_type: WorkflowType,
    pub steps: Vec<WorkflowStep>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub causal_chain: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkflowType {
    Nextflow,
    Jupyter,
    Snakemake,
    Custom(String),
}

impl ScientificWorkflowTrace {
    pub fn new(workflow_name: &str, workflow_type: WorkflowType) -> Self {
        let trace_id = blake3::hash(
            format!("{}:{}", workflow_name, Utc::now().timestamp_millis()).as_bytes()
        ).to_string();

        Self {
            trace_id,
            workflow_name: workflow_name.to_string(),
            workflow_type,
            steps: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            causal_chain: String::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn add_step(&mut self, mut step: WorkflowStep) -> Result<()> {
        if self.steps.iter().any(|s| s.id == step.id) {
            return Err(DesciError::PluginValidation(
                format!("Duplicate step id: {}", step.id)
            ));
        }

        step.hash = Some(step.compute_hash());
        self.steps.push(step);
        self.recompute_causal_chain();
        self.updated_at = Utc::now();
        Ok(())
    }

    fn recompute_causal_chain(&mut self) {
        let mut chain = format!("{}:{}", self.trace_id, self.workflow_name);

        for step in &self.steps {
            let step_hash = step.hash.as_deref().unwrap_or("");
            chain = blake3::hash(
                format!("{}:{}", chain, step_hash).as_bytes()
            ).to_string();
        }

        self.causal_chain = chain;
    }

    pub fn verify(&self) -> bool {
        let mut chain = format!("{}:{}", self.trace_id, self.workflow_name);

        for step in &self.steps {
            let expected_hash = step.compute_hash();
            if step.hash.as_deref() != Some(&expected_hash) {
                info!(
                    step = %step.id,
                    expected = %expected_hash,
                    actual = ?step.hash,
                    "Step hash mismatch"
                );
                return false;
            }

            let step_hash = step.hash.as_deref().unwrap_or("");
            chain = blake3::hash(
                format!("{}:{}", chain, step_hash).as_bytes()
            ).to_string();
        }

        let verified = chain == self.causal_chain;
        if !verified {
            info!(
                expected = %chain,
                actual = %self.causal_chain,
                "Causal chain mismatch"
            );
        }
        verified
    }

    pub fn get_step(&self, id: &str) -> Option<&WorkflowStep> {
        self.steps.iter().find(|s| s.id.as_str() == id)
    }

    pub fn get_step_mut(&mut self, id: &str) -> Option<&mut WorkflowStep> {
        self.steps.iter_mut().find(|s| s.id.as_str() == id)
    }

    pub fn completed_count(&self) -> usize {
        self.steps.iter().filter(|s| matches!(s.status, StepStatus::Completed)).count()
    }

    pub fn total_count(&self) -> usize {
        self.steps.len()
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

pub trait NextflowTraceExt {
    fn from_nextflow_execution(execution_json: &str) -> Result<ScientificWorkflowTrace>;
}

impl NextflowTraceExt for ScientificWorkflowTrace {
    fn from_nextflow_execution(execution_json: &str) -> Result<ScientificWorkflowTrace> {
        let v: serde_json::Value = serde_json::from_str(execution_json)
            .map_err(|e| DesciError::Parse(format!("Invalid Nextflow JSON: {}", e)))?;

        let workflow_name = v["workflow"]
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("unnamed");

        let mut trace = ScientificWorkflowTrace::new(workflow_name, WorkflowType::Nextflow);

        if let Some(processes) = v["processes"].as_array() {
            for (i, proc) in processes.iter().enumerate() {
                let name = proc.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");

                let mut step = WorkflowStep::new(
                    format!("nf-{}", i),
                    name,
                    proc.get("process")
                        .and_then(|p| p.as_str())
                        .unwrap_or("unknown"),
                );

                step.start();

                if let Some(status) = proc.get("status").and_then(|s| s.as_str()) {
                    match status {
                        "COMPLETED" => {
                            let outputs: Vec<String> = proc.get("out")
                                .and_then(|o| o.as_array())
                                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                                .unwrap_or_default();
                            step.complete(outputs);
                        }
                        "FAILED" => {
                            let err = proc.get("error")
                                .and_then(|e| e.as_str())
                                .unwrap_or("Unknown error");
                            step.fail(err);
                        }
                        _ => {}
                    }
                }

                trace.add_step(step)?;
            }
        }

        Ok(trace)
    }
}
