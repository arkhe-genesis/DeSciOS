/**
 * TopoMAS v9.2 — Space Dashboard Visualizer v2.0 (WEBGL Fixed & Quaternion)
 * ============================================================================
 */

let data;
let t = 0;
let selectedIndex = 0;
let isDragging = false;
let lastMX, lastMY;
let currentView = 0;
let animSpeed = 1.0;

const CONFIG = {
    colors: {
        "Trivial": [0, 0, 80], "Topological_Insulator": [200, 80, 80],
        "Topological_Semimetal": [120, 80, 80], "TI": [200, 80, 80], "TSM": [120, 80, 80],
    },
    weights: { radiation_hardness: 0.25, vacuum_stability: 0.25, thermal_cycling: 0.20, weight_efficiency: 0.15, synthesizability: 0.15 }
};

// === Quaternion Class for p5.js (Hamiltonian implementation) ===
class Quaternion {
    constructor(w, x, y, z) { this.w = w; this.x = x; this.y = y; this.z = z; }

    static fromEuler(ax, ay, az) {
        let cy = Math.cos(az * 0.5), sy = Math.sin(az * 0.5);
        let cp = Math.cos(ay * 0.5), sp = Math.sin(ay * 0.5);
        let cr = Math.cos(ax * 0.5), sr = Math.sin(ax * 0.5);
        return new Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    slerp(qb, t) {
        let qa = this, omega, cosom = qa.w*qb.w + qa.x*qb.x + qa.y*qb.y + qa.z*qb.z;
        if (cosom < 0.0) { cosom = -cosom; qb = new Quaternion(-qb.w, -qb.x, -qb.y, -qb.z); }
        if ((1.0 - cosom) > 1e-6) {
            omega = Math.acos(cosom);
            let sinom = Math.sin(omega);
            return new Quaternion(
                Math.sin((1-t)*omega)/sinom * qa.w + Math.sin(t*omega)/sinom * qb.w,
                Math.sin((1-t)*omega)/sinom * qa.x + Math.sin(t*omega)/sinom * qb.x,
                Math.sin((1-t)*omega)/sinom * qa.y + Math.sin(t*omega)/sinom * qb.y,
                Math.sin((1-t)*omega)/sinom * qa.z + Math.sin(t*omega)/sinom * qb.z
            );
        } else return new Quaternion(qa.w*(1-t)+qb.w*t, qa.x*(1-t)+qb.x*t, qa.y*(1-t)+qb.y*t, qa.z*(1-t)+qb.z*t);
    }

    multiply(qb) {
        return new Quaternion(
            this.w * qb.w - this.x * qb.x - this.y * qb.y - this.z * qb.z,
            this.w * qb.x + this.x * qb.w + this.y * qb.z - this.z * qb.y,
            this.w * qb.y - this.x * qb.z + this.y * qb.w + this.z * qb.x,
            this.w * qb.z + this.x * qb.y - this.y * qb.x + this.z * qb.w
        );
    }

    toMatrix() {
        let x=this.x, y=this.y, z=this.z, w=this.w;
        return [
            1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y, 0,
            2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x, 0,
            2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y, 0,
            0, 0, 0, 1
        ];
    }
}

function preload() {
    try { data = loadJSON('space_dashboard.json'); }
    catch (e) { data = getExampleData(); }
}

function getExampleData() {
    return {
        space_scores: [
            { id: "m1", formula: "SnSe0.9Te0.1", radiation_hardness: 0.9, vacuum_stability: 0.95, thermal_cycling: 0.9, weight_efficiency: 0.7, synthesizability: 0.85, overall_score: 0.94, class: "TI" },
            { id: "m2", formula: "Bi2Se3", radiation_hardness: 0.65, vacuum_stability: 0.6, thermal_cycling: 0.7, weight_efficiency: 0.8, synthesizability: 0.8, overall_score: 0.76, class: "TI" },
            { id: "m3", formula: "WTe2", radiation_hardness: 0.8, vacuum_stability: 0.7, thermal_cycling: 0.7, weight_efficiency: 0.9, synthesizability: 0.6, overall_score: 0.82, class: "TSM" }
        ],
        structures: [{ id: "m1", formula: "SnSe0.9Te0.1", atoms: [ {element:"Sn",x:0,y:0,z:0,radius:0.8}, {element:"Se",x:1,y:0,z:0,radius:0.6}, {element:"Te",x:0,y:1,z:0,radius:0.7} ], bonds: [[0,1],[0,2],[1,2]] }]
    };
}

function setup() {
    createCanvas(windowWidth, windowHeight, WEBGL);
    colorMode(HSB, 360, 100, 100, 100);
    textFont('monospace');

    let btn = createButton('Alternar Visão (Pareto / Estrutura / Radar)');
    btn.position(10, 10);
    btn.mousePressed(() => currentView = (currentView + 1) % 3);
}

function draw() {
    background(220, 20, 8);
    t += 0.01 * animSpeed;

    // Renderiza conteúdo 3D/2D principal
    push();
    if (currentView === 0) drawParetoFront();
    else if (currentView === 1) drawCrystalStructure();
    else if (currentView === 2) drawRadarChart();
    pop();

    // Desenha UI 2D sobreposta (Correção principal do WEBGL)
    drawUI();
}

// =============================================================================
// SISTEMA UI 2D SOBREPOSTO (WEBGL HACK)
// =============================================================================
function drawUI() {
    push();
    // Move o ponto 0,0 para o canto superior esquerdo e traz para frente
    translate(-width / 2, -height / 2, 500);
    noStroke();

    // Fundo do Header
    fill(220, 50, 10, 80);
    rect(0, 0, width, 50);

    // Título
    fill(0, 0, 100);
    textSize(18);
    textAlign(LEFT, CENTER);
    text('TopoMAS v9.1 — Space Dashboard', 180, 25);

    // Info atual
    fill(0, 0, 70);
    textSize(12);
    text(`Modo: ${['Pareto 2D', 'Estrutura 3D', 'Radar Comparativo'][currentView]}`, width - 250, 25);
    pop();

    drawLegend();
}

function drawLegend() {
    push();
    translate(-width/2 + width - 180, -height/2 + 70, 500);
    fill(220, 50, 15, 90);
    stroke(0, 0, 40);
    rect(0, 0, 160, 110, 8);

    fill(0, 0, 100); noStroke(); textSize(12); textAlign(LEFT, TOP);
    text('Legenda', 15, 15);

    let items = [{l:'TI', h:200}, {l:'TSM', h:120}, {l:'Trivial', h:0}];
    for(let i=0; i<items.length; i++) {
        fill(items[i].h, 80, 80); noStroke();
        circle(25, 45 + i*22, 10);
        fill(0, 0, 80); textSize(11);
        text(items[i].l, 40, 39 + i*22);
    }
    pop();
}

// =============================================================================
// 1. FRONTEIRA DE PARETO (Mapeado para WEBGL corretamente)
// =============================================================================
function drawParetoFront() {
    let scores = data.space_scores || [];
    if(!scores.length) return;

    let s = min(width, height) * 0.6;

    // Eixos
    stroke(0, 0, 40); strokeWeight(1);
    line(-s/2, s/2, s/2, s/2); // X
    line(-s/2, s/2, -s/2, -s/2); // Y

    // Labels
    push(); translate(s/2 - 80, s/2 + 25, 0); fill(0,0,70); noStroke(); textSize(11);
    text('Topological Score', 0, 0); pop();

    push(); translate(-s/2 - 50, 0, 0); rotate(-PI/2); fill(0,0,70); noStroke(); textSize(11);
    text('Space Robustness', 0, 0); pop();

    // Pontos
    for (let i = 0; i < scores.length; i++) {
        let p = scores[i];
        // Mapeamento fake para demonstração (usa overall_score e radiation)
        let x = -s/2 + (p.overall_score || 0.5) * s;
        let y = s/2 - (p.radiation_hardness || 0.5) * s;
        let h = CONFIG.colors[p.class] ? CONFIG.colors[p.class][0] : 200;

        // Brilho animado
        let r = 10 + 4 * sin(t * 2 + i);
        for(let j=3; j>0; j--) {
            stroke(h, 80, 80, 10/j); strokeWeight(r*j); point(x, y);
        }

        stroke(h, 80, 100); strokeWeight(2); noFill();
        circle(x, y, r*2);

        if(i === selectedIndex) { stroke(0,0,100); strokeWeight(3); circle(x, y, r*3); }

        push(); translate(x+15, y-10, 0); fill(h, 80, 100); noStroke(); textSize(11);
        text(p.formula, 0, 0); pop();

        if(dist(mouseX-width/2, mouseY-height/2, x, y) < 20) selectedIndex = i;
    }
}

// =============================================================================
// 2. ESTRUTURA CRISTALINA 3D (Com Orbit Control Nativo via Quaternion)
// =============================================================================

let quatTarget = new Quaternion(1, 0, 0, 0); // Target orientation
let quatCurrent = new Quaternion(1, 0, 0, 0); // Smoothly interpolated orientation

function mouseDragged() {
    // Update target based on mouse drag (without gimbal lock)
    let dx = (mouseX - pmouseX) * 0.01;
    let dy = (mouseY - pmouseY) * 0.01;
    let qx = new Quaternion(cos(dy/2), sin(dy/2), 0, 0);
    let qy = new Quaternion(cos(dx/2), 0, sin(dx/2), 0);
    quatTarget = qx.multiply(qy.multiply(quatTarget));
}

function drawCrystalStructure() {
    let structs = data.structures || [];
    if(!structs.length) return;
    let s = structs[selectedIndex] || structs[0];

    if (!isDragging) {
        // Auto-rotação
        let dx = 0.005;
        let qy = new Quaternion(cos(dx/2), 0, sin(dx/2), 0);
        quatTarget = qy.multiply(quatTarget);
    }

    // Smooth Slerp from current to target
    quatCurrent = quatCurrent.slerp(quatTarget, 0.08);
    let m = quatCurrent.toMatrix();
    applyMatrix(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);

    let sc = 150;

    // Ligações
    stroke(0, 0, 50, 40); strokeWeight(2);
    for (let b of (s.bonds || [])) {
        let a1 = s.atoms[b[0]], a2 = s.atoms[b[1]];
        if(a1 && a2) line(a1.x*sc, a1.y*sc, a1.z*sc, a2.x*sc, a2.y*sc, a2.z*sc);
    }

    // Átomos
    for (let a of s.atoms) {
        let h = getElementColor(a.element);
        push();
        translate(a.x*sc, a.y*sc, a.z*sc);
        fill(h, 80, 80, 90); stroke(h, 80, 100); strokeWeight(1);
        sphere((a.radius || 0.5) * 25);
        pop();
    }
}

function getElementColor(el) { return (el === "Sn" || el === "Sb") ? 50 : (el === "Se" ? 180 : 60); }

// =============================================================================
// 3. GRÁFICO RADAR MULTI-CANDIDATO
// =============================================================================
function drawRadarChart() {
    let scores = data.space_scores || [];
    if(!scores.length) return;

    let rad = min(width, height) * 0.3;
    let cats = ['radiation_hardness', 'vacuum_stability', 'thermal_cycling', 'weight_efficiency', 'synthesizability'];
    let labels = ['Radiação', 'Vácuo', 'Ciclos T.', 'Peso', 'Sintese'];
    let n = cats.length;

    // Grades e Eixos
    stroke(0, 0, 30); strokeWeight(0.5);
    for(let r=0.2; r<=1.0; r+=0.2) {
        noFill(); beginShape();
        for(let i=0; i<=n; i++) { let a = -PI/2 + (i/n)*TWO_PI; vertex(cos(a)*rad*r, sin(a)*rad*r); }
        endShape();
    }
    for(let i=0; i<n; i++) {
        let a = -PI/2 + (i/n)*TWO_PI;
        line(0, 0, cos(a)*rad, sin(a)*rad);
        push(); translate(cos(a)*(rad+30), sin(a)*(rad+30), 0); fill(0,0,80); noStroke(); textSize(12); textAlign(CENTER, CENTER);
        text(labels[i], 0, 0); pop();
    }

    // Desenha TODOS os candidatos (fantasmas) e o selecionado em destaque
    for (let c=0; c<scores.length; c++) {
        let p = scores[c];
        let h = CONFIG.colors[p.class] ? CONFIG.colors[p.class][0] : 200;
        let isSelected = (c === selectedIndex);

        fill(h, 80, 80, isSelected ? 30 : 5);
        stroke(h, 80, 80, isSelected ? 100 : 20);
        strokeWeight(isSelected ? 3 : 1);

        beginShape();
        for(let i=0; i<n; i++) {
            let a = -PI/2 + (i/n)*TWO_PI;
            let v = p[cats[i]] || 0;
            vertex(cos(a)*rad*v, sin(a)*rad*v);
        }
        endShape(CLOSE);

        // Label do selecionado
        if(isSelected) {
            push(); translate(0, rad + 60, 0); fill(0,0,100); noStroke(); textSize(16); textAlign(CENTER, CENTER);
            text(p.formula, 0, 0); pop();
        }
    }
}

// =============================================================================
// INTERAÇÕES
// =============================================================================
function mousePressed() { isDragging = true; lastMX = mouseX; lastMY = mouseY; }
function mouseReleased() { isDragging = false; }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
