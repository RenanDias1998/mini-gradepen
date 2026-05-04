import sys
import cv2
from omr_core import run_omr, OMRException


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'prova.jpg'
    img = cv2.imread(path)
    if img is None:
        print(f'ERRO: nao foi possivel abrir {path}')
        return 2

    try:
        result = run_omr(img)
    except OMRException as exc:
        print(f'ERRO OMR: {exc}')
        return 3

    print('OK: leitura concluida')
    print('Respostas:', ' '.join(result.answers))
    print('Status:', ', '.join(q.status for q in result.questions))
    print('Confiancas:', ', '.join(f'{q.confidence:.4f}' for q in result.questions))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
