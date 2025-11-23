import subprocess
import sys
import os
import platform

# Configuration
VENV_NAME = ".DAG_env"
REQUIREMENTS_FILE = "requirements.txt"

def run_command(command, description):
    """Exécute une commande système avec gestion d'erreur."""
    print(f"--- {description} ---")
    try:
        # On utilise sys.executable pour être sûr d'utiliser le python actuel
        # cela contourne ton problème de PATH "uv not recognized"
        subprocess.run(command, check=True)
        print("✅ Succès.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de : {description}")
        print(f"Détail : {e}")
        sys.exit(1)

def main():
    print(f"🚀 Démarrage de l'installation du projet...\n")

    # 1. Vérifier/Installer UV
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True, check=True)
        print("✅ uv est déjà installé.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ uv non trouvé. Installation en cours via pip...")
        run_command([sys.executable, "-m", "pip", "install", "uv"], "Installation de uv")

    # 2. Créer l'environnement virtuel
    if not os.path.exists(VENV_NAME):
        # On utilise "python -m uv venv" pour éviter ton erreur PowerShell
        cmd = [sys.executable, "-m", "uv", "venv", VENV_NAME]
        run_command(cmd, f"Création du venv '{VENV_NAME}'")
    else:
        print(f"ℹ️ Le dossier {VENV_NAME} existe déjà. On continue.\n")

    # 3. Installer les dépendances si requirements.txt existe
    if os.path.exists(REQUIREMENTS_FILE):
        cmd = [sys.executable, "-m", "uv", "pip", "install", "-r", REQUIREMENTS_FILE]
        run_command(cmd, "Installation des modules (via uv)")
    else:
        print(f"ℹ️ Aucun fichier {REQUIREMENTS_FILE} trouvé. Pas de modules à installer.\n")

    # 4. Afficher comment activer
    print("🎉 Installation terminée !")
    print("Pour activer l'environnement, lance cette commande :\n")
    
    if platform.system() == "Windows":
        print(f"    {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"    source {VENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()