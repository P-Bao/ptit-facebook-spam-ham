"""Tạo gói truyền nội bộ sang Colab; không chứa cookie, token hay dữ liệu ngoài project."""
from pathlib import Path
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'.work/colab_transfer/ptit_vibert_bundle.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    files=[root/'pipeline/__init__.py',root/'pipeline/modeling/__init__.py',
           root/'pipeline/modeling/training_utils.py',root/'pipeline/presentation/__init__.py',
           root/'pipeline/presentation/notebook_display.py',root/'requirements/modern-colab.txt',
           root/'data/final/labeled_v7.csv',root/'artifacts/spamham/test_predictions.csv',
           root/'artifacts/spamham/metrics.csv',root/'artifacts/spamham/run_manifest.json']
    files+=sorted((root/'pipeline/modern').glob('*.py'))
    files+=sorted((root/'pipeline/modern').glob('*.json'))
    files += [root/'pipeline/modern/README.md', root/'requirements/modern.txt',
              root/'docs/guides/HUONG_DAN_VIBERT_COLAB.md']
    if (root/'notebooks/03_vibert.ipynb').exists():files.append(root/'notebooks/03_vibert.ipynb')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in files:archive.write(file,file.relative_to(root).as_posix())
    print(f'{output.relative_to(root)}: {len(files)} files, {output.stat().st_size/1024/1024:.2f} MiB')


if __name__=='__main__':main()
