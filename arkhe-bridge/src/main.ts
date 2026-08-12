import { Plugin, TFile, Notice } from 'obsidian';
import * as crypto from 'crypto';

export default class ArkheBridgePlugin extends Plugin {
  async onload() {
    // 1. Comando: Gerar hash da nota atual
    this.addCommand({
      id: 'generate-note-hash',
      name: 'Generate SHA-256 hash for current note',
      callback: () => this.generateHashForCurrentNote(),
    });

    // 2. Comando: Validar selo
    this.addCommand({
      id: 'validate-seal',
      name: 'Validate seal against hash',
      callback: () => this.validateSealCommand(),
    });

    // 3. Evento: Atualizar hash ao salvar
    this.registerEvent(
      this.app.vault.on('modify', (file) => {
        if (file instanceof TFile && file.extension === 'md') {
          this.updateHashInFrontmatter(file);
        }
      })
    );

    // 4. API exposta para outros plugins
    (this.app as any).arkhe = {
      getNoteHash: (file: TFile) => this.getNoteHash(file),
      validateSeal: (file: TFile) => this.validateSeal(file),
    };
  }

  async generateHashForCurrentNote() {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice('No active file');
      return;
    }
    const content = await this.app.vault.read(file);
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    await this.app.fileManager.processFrontMatter(file, (fm) => {
      fm.hash = hash;
      fm.timestamp = new Date().toISOString();
    });
    new Notice(`Hash gerado: ${hash.slice(0, 8)}...`);
  }

  async updateHashInFrontmatter(file: TFile) {
    // Atualiza hash se o conteúdo mudou
    const content = await this.app.vault.read(file);
    const newHash = crypto.createHash('sha256').update(content).digest('hex');
    await this.app.fileManager.processFrontMatter(file, (fm) => {
      if (fm.hash !== newHash) {
        fm.hash = newHash;
        fm.modified = new Date().toISOString();
      }
    });
  }

  getNoteHash(file: TFile): string | null {
    const cache = this.app.metadataCache.getFileCache(file);
    return cache?.frontmatter?.hash || null;
  }

  validateSeal(file: TFile): boolean {
    const cache = this.app.metadataCache.getFileCache(file);
    const seal = cache?.frontmatter?.selo;
    const hash = cache?.frontmatter?.hash;
    if (!seal || !hash) return false;
    // Verificação simples: selo contém hash?
    return seal.includes(hash.slice(0, 8));
  }

  validateSealCommand() {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice('No active file');
      return;
    }
    const isValid = this.validateSeal(file);
    if (isValid) {
        new Notice('Selo válido!');
    } else {
        new Notice('Selo inválido ou ausente!');
    }
  }
}
