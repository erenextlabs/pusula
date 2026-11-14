# Contributing to Pusula

[Türkçe](#pusulaya-katkı)

Contributions to Pusula are made through pull requests. The rules below apply to every contributor.

## Development setup

```bash
git clone https://github.com/erenextlabs/pusula.git
cd pusula
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check pusula tests
pytest -q
```

The tests run on CPU and do not download any model. A GPU is not required for development.

## Branches

- `main` holds the most recent public release. Each release is marked with a tag (`v1.0`, `v2.0`).
- Development takes place on the active version branch (`vX.Y`, the most recent one).
- Work is carried out on a separate branch named `<github-username>/<topic>`, created from the active version branch.
- A pull request is opened into the active version branch, not into `main`.

## Pull requests

- Each pull request is limited to a single, clearly described change.
- `ruff check pusula tests` and `pytest -q` are expected to pass before a review is requested.
- New behaviour is covered by tests that run on CPU.
- Pull requests are reviewed and merged by the maintainers. Before a merge, the tests are run again and
  the changes are checked for personal data.

## What must not be committed

- Credentials of any kind: `.env` files, API keys, tokens, passwords, server addresses and ports.
- Personal data of any kind.
- Real customer messages, or any text, label, log or report derived from them.
- Training data and model weights. Model weights are distributed through Hugging Face only.

Examples in documentation, tests and code are synthetic.

## License

This repository is licensed under the [Apache License 2.0](LICENSE). Under Section 5 of that license, a
contribution submitted for inclusion is licensed under the same terms. The copyright notice in
[`NOTICE`](NOTICE) is not to be changed in a pull request.

---

# Pusula'ya katkı

Pusula'ya katkılar değişiklik isteği (pull request) yoluyla yapılmaktadır. Aşağıdaki kurallar tüm katkıda
bulunanlar için geçerlidir.

## Geliştirme ortamı

```bash
git clone https://github.com/erenextlabs/pusula.git
cd pusula
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check pusula tests
pytest -q
```

Testler CPU üzerinde çalışmakta ve herhangi bir model indirmemektedir. Geliştirme için GPU gerekmemektedir.

## Dallar

- `main` dalında en son herkese açık sürüm bulunmaktadır. Her sürüm bir etiketle işaretlenmektedir (`v1.0`, `v2.0`).
- Geliştirme, etkin sürüm dalında (`vX.Y`, en güncel olanı) yürütülmektedir.
- Çalışmalar, etkin sürüm dalından açılan ve `<github-kullanıcı-adı>/<konu>` biçiminde adlandırılan ayrı bir dalda yapılmaktadır.
- Değişiklik isteği `main` dalına değil, etkin sürüm dalına açılmaktadır.

## Değişiklik istekleri

- Her değişiklik isteği tek ve açıkça tanımlanmış bir değişiklikle sınırlandırılmaktadır.
- İnceleme istenmeden önce `ruff check pusula tests` ve `pytest -q` komutlarının başarıyla tamamlanması beklenmektedir.
- Yeni davranışlar, CPU üzerinde çalışan testlerle kapsanmaktadır.
- Değişiklik istekleri proje yürütücüleri tarafından incelenmekte ve birleştirilmektedir. Birleştirme öncesinde
  testler yeniden çalıştırılmakta ve değişiklikler kişisel veri açısından denetlenmektedir.

## Depoya eklenmemesi gerekenler

- Her türlü kimlik bilgisi: `.env` dosyaları, API anahtarları, erişim belirteçleri, şifreler, sunucu adresleri ve portlar.
- Her türlü kişisel veri.
- Gerçek müşteri mesajları ya da bu mesajlardan türetilmiş metin, etiket, kayıt veya rapor.
- Eğitim verisi ve model ağırlıkları. Model ağırlıkları yalnızca Hugging Face üzerinden dağıtılmaktadır.

Belgelerde, testlerde ve kodda yer alan örnekler sentetiktir.

## Lisans

Bu depo [Apache Lisansı 2.0](LICENSE) ile lisanslanmıştır. Bu lisansın 5. bölümü uyarınca, dahil edilmek
üzere sunulan katkılar aynı koşullarla lisanslanmış sayılmaktadır. [`NOTICE`](NOTICE) dosyasındaki telif
bildirimi bir değişiklik isteğiyle değiştirilmemektedir.
