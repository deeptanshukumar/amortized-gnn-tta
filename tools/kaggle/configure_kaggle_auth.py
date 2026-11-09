"""Save a Kaggle token locally without echoing it or changing project files."""
from getpass import getpass
import os
from pathlib import Path
import tempfile


def main():
    directory = Path.home() / '.kaggle'
    if directory.is_symlink():
        raise RuntimeError('Inspect the symlinked Kaggle configuration directory first')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = directory / 'access_token'
    if destination.exists():
        raise FileExistsError('An access token already exists; preserve it and inspect before replacement')
    token = getpass('Paste your replacement Kaggle API token (hidden): ').strip()
    if not token.startswith('KGAT_') or len(token) < 10 or any(c.isspace() for c in token):
        raise ValueError('Expected a Kaggle access token beginning KGAT_; value was not printed')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False, encoding='utf-8') as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(token + '\n')
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        token = None
    print('Credential saved privately outside the repository. The token was not printed.')
    print("Check access with: uvx --with 'six>=1.16,<2' --from kaggle==2.2.5 kaggle quota")


if __name__ == '__main__':
    main()
