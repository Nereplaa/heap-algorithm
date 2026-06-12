"""
word_freq_heap.py
==================

Bir TXT dosyasındaki kelime frekanslarını sayar ve şu şekilde sıralı yazdırır:

    1. Önce kelimenin İLK HARFİNE göre alfabetik olarak (A -> Z), ardından
    2. Aynı ilk harfe sahip kelimeler için FREKANSA (tekrar sayısına) göre
       azalan sırada (en yüksek sayı önce).

Yukarıdaki sıralamanın tamamı, ödevin gerektirdiği tam olarak iki anahtara
göre sıralama yapan, elle yazılmış TEK BİR ikili (binary) heap (aşağıdaki
`DualKeyHeap` sınıfı) tarafından üretilir:

    BİRİNCİL ANAHTAR : kelimenin ilk harfi      -> artan sıra (A < B < ... < Z)
    İKİNCİL ANAHTAR  : kelimenin tekrar sayısı  -> azalan sıra (SADECE aynı
                                                    ilk harfe sahip kelimeler
                                                    arasındaki eşitlikleri
                                                    bozmak için kullanılır)

Heap'i oluşturmak veya sıralamak için `heapq`, `sorted()`, `PriorityQueue`
veya başka herhangi bir hazır öncelik kuyruğu / sıralama aracı KULLANILMAZ.
Heap, klasik ders kitabı heap'i gibi, dizi (array) tabanlı bir ikili ağaç
olarak kullanılan sade bir Python listesidir.

Algoritmanın tam açıklaması, karmaşıklık analizi, Kocaeli/Konya/... girdisiyle
çözülmüş örnek ve uç durumlar için REPORT.md dosyasına bakın.

KULLANIM
--------
    python word_freq_heap.py <dosya_yolu.txt>            normal çalıştırma
    python word_freq_heap.py <dosya_yolu.txt> --debug    her kelimeden sonra
                                                          heap durumunu gösterir
    python word_freq_heap.py --demo                      ödevdeki
                                                          Kocaeli/Konya/...
                                                          örneğini adım adım
                                                          çalıştırır
    python word_freq_heap.py                             dosya yolu sormak
                                                          için
"""

from __future__ import annotations

import argparse
import re
import sys

# Windows'taki varsayılan konsol kod sayfası (örn. cp1252) Türkçe
# karakterleri (ı, ş, ğ, ç, ö, ü, İ) içermez ve print() sırasında
# UnicodeEncodeError fırlatır; çıkışı UTF-8'e zorlayarak bunu önlüyoruz.
sys.stdout.reconfigure(encoding="utf-8")


# ======================================================================
#  WordNode
# ======================================================================
class WordNode:
    """
    Heap içinde saklanan tek bir eleman: normalize edilmiş bir kelime ve
    o kelimenin şimdiye kadar kaç kez görüldüğü.

    Düz bir liste/demet (list/tuple) yerine küçük bir sınıf kullanmak,
    aşağıdaki heap kodunun "node[0]" / "node[1]" yerine "node.word" /
    "node.count" şeklinde okunmasını sağlar; bu da kodu takip etmeyi
    çok daha kolay hale getirir.
    """

    __slots__ = ("word", "count")

    def __init__(self, word: str, count: int = 1) -> None:
        self.word = word
        self.count = count

    def __repr__(self) -> str:
        return f"{self.word}({self.count})"


# ======================================================================
#  DualKeyHeap  -- ödevin gerektirdiği özel heap
# ======================================================================
class DualKeyHeap:
    """
    Sade bir Python listesi olarak saklanan bir ikili (binary) heap
    (dizi tabanlı, tam olarak klasik ders kitabı heap'i gibi: ``i``
    indeksinin çocukları ``2*i + 1`` ve ``2*i + 2`` konumlarında,
    ``i``'nin ebeveyni de ``(i - 1) // 2`` konumunda yer alır).

    SIRALAMA KURALI ("çift anahtar / dual key")
    --------------------------------------------
    Köke daha yakın oturan düğüm, ÖNCELİĞİ DAHA YÜKSEK olan düğümdür.
    Bu öncelik şöyle tanımlanır:

        1. Kelimesinin İLK HARFİ alfabetik olarak daha küçük olan
           düğüm (A, B'yi yener; B, C'yi yener; ... Z'ye kadar).
        2. Eğer iki düğüm de AYNI harfle başlıyorsa, tekrar SAYISI
           DAHA BÜYÜK olan düğüm kazanır.

    Bu kural sayesinde, heap'in kökünü tekrar tekrar kaldırmak
    (``extract_top``) düğümleri ödevin istediği nihai çıktı sırasıyla,
    yani A -> Z şeklinde ve ilk harfte eşitlik olduğunda en yüksek
    sayıdan en düşüğe doğru verir.

    HIZLI "ZATEN VAR MI?" SORGUSU
    --------------------------------
    Sade bir heap, O(1) zamanda yalnızca kökü hakkında bilgi verebilir;
    rastgele bir kelimeyi bulmak normalde O(n) sürer. "Bir kelimeyi
    eklemeden önce zaten var olup olmadığını kontrol et, varsa sayısını
    artır ve heap'i yeniden düzenle" gereksinimini karşılamak için bu
    heap, ``word -> dizideki konum`` eşlemesini tutan yardımcı bir
    ``self._index`` sözlüğü kullanır. İki düğüm her yer değiştirdiğinde,
    ``self._index`` içindeki ilgili iki kayıt da güncellenir; böylece
    sözlük her zaman her kelimenin gerçek konumunu yansıtır. Bu sayede
    "var mı?" kontrolü ve "güncellenecek düğümü bul" adımı O(1) işlemlere
    dönüşür.
    """

    def __init__(self) -> None:
        # Heap'in kendisi: WordNode nesnelerinden oluşan, dizi tabanlı ikili ağaç.
        self._data: list[WordNode] = []

        # word -> bu kelimenin düğümünün self._data içindeki indeksi.
        # Her yer değiştirmede güncel tutulur, böylece her kelimeyi O(1)'de buluruz.
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Temel sorgulama metotları
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        """Heap şu anda hiç kelime içermiyorsa True döner."""
        return len(self._data) == 0

    def contains(self, word: str) -> bool:
        """``word`` heap içinde zaten bir düğüme sahipse True döner."""
        return word in self._index

    def to_list(self) -> list[tuple[str, int]]:
        """
        Heap'in alttaki dizisinin anlık görüntüsünü ``(word, count)``
        demetlerinden oluşan bir liste olarak, dizi sırasıyla (0. indeks
        = kök) döndürür. Hata ayıklama / yazdırma için kullanışlıdır.
        """
        return [(node.word, node.count) for node in self._data]

    # ------------------------------------------------------------------
    # Çift anahtar karşılaştırması: bu tek metot HER İKİ anahtarı da kodlar.
    # ------------------------------------------------------------------
    def _has_priority(self, i: int, j: int) -> bool:
        """
        ``i`` indeksindeki düğümün önceliği ``j`` indeksindeki düğümden
        KESİN OLARAK DAHA YÜKSEKSE True döner; yani ``i``, köke ``j``'den
        daha yakın oturmaya hak kazanır.

        Çift anahtar (dual-key) kuralı:
          - BİRİNCİL  : ilk harfi daha küçük olan -> önceliği daha yüksek
          - İKİNCİL   : (sadece ilk harfler eşitse) sayısı daha büyük
                        olan -> önceliği daha yüksek
        """
        a, b = self._data[i], self._data[j]

        letter_a, letter_b = a.word[0], b.word[0]
        if letter_a != letter_b:
            # Birincil anahtar: alfabetik sıra, A en "küçük" (köke en yakın) harftir.
            return letter_a < letter_b

        # İlk harfler aynı -> ikincil anahtar belirler: sayısı büyük olan kazanır.
        return a.count > b.count

    # ------------------------------------------------------------------
    # Dahili yardımcılar: swap + sift (heapify) yukarı / aşağı
    # ------------------------------------------------------------------
    def _swap(self, i: int, j: int) -> None:
        """Dizideki iki düğümü yer değiştirir VE self._index'i güncel tutar."""
        self._data[i], self._data[j] = self._data[j], self._data[i]
        self._index[self._data[i].word] = i
        self._index[self._data[j].word] = j

    def _sift_up(self, i: int) -> None:
        """
        ``i`` indeksindeki düğümü, ebeveyninden daha yüksek önceliğe
        sahip olduğu sürece ağaçta YUKARI taşır. Bu, hem dizinin sonuna
        eklenen yepyeni düğümler hem de sayısı az önce artırılan mevcut
        düğümler için kullanılan standart "yukarı kabarcıklanma"
        (bubble up) adımıdır.
        """
        while i > 0:
            parent = (i - 1) // 2
            if self._has_priority(i, parent):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """
        ``i`` indeksindeki düğümü, her iki çocuğu da daha düşük (veya eşit)
        önceliğe sahip olana kadar, önceliği daha yüksek olan çocuğuyla
        tekrar tekrar yer değiştirerek ağaçta AŞAĞI taşır. Kök kaldırıldıktan
        ve dizinin son elemanıyla değiştirildikten sonra kullanılır.
        """
        n = len(self._data)
        while True:
            best = i
            left, right = 2 * i + 1, 2 * i + 2

            if left < n and self._has_priority(left, best):
                best = left
            if right < n and self._has_priority(right, best):
                best = right

            if best == i:
                break

            self._swap(i, best)
            i = best

    # ------------------------------------------------------------------
    # Genel işlem #1: yeni bir kelime ekle VEYA mevcut olanı artır
    # ------------------------------------------------------------------
    def insert_or_increment(self, word: str) -> None:
        """
        Dosyadan okunan bir kelimeyi işler.

          * Kelime heap'te ZATEN VARSA: sayısını bir artırır ve heap
            özelliğinin yeniden sağlanması için yeniden heapify eder
            (yukarı kabarcıklandırır - sift up).
          * Kelime heap'te HENÜZ YOKSA: count = 1 olan yeni bir yaprak
            (leaf) olarak ekler, ardından doğru konumuna kabarcıklandırır.

        Artırma işleminde neden SADECE `_sift_up`'ın yeterli olduğuna dair not:
        Bir düğümün sayısını artırmak, önceliğini ASLA DÜŞÜREMEZ (çift
        anahtar kuralı, sayıları yalnızca aynı ilk harfe sahip düğümler
        arasında karşılaştırır ve daha yüksek bir sayı her zaman daha
        düşük bir sayıdan en az o kadar iyidir). Heap, artırmadan *önce*
        geçerli olduğu için, sonrasında ortaya çıkabilecek tek ihlal "bu
        düğüm artık ebeveyninden DAHA YÜKSEKTE olmayı hak ediyor" şeklinde
        olabilir - bu da tam olarak `_sift_up`'ın düzelttiği durumdur.
        `_sift_down`'a asla gerek kalmaz.
        """
        if word in self._index:
            i = self._index[word]
            self._data[i].count += 1
            self._sift_up(i)
        else:
            self._data.append(WordNode(word, 1))
            i = len(self._data) - 1
            self._index[word] = i
            self._sift_up(i)

    # ------------------------------------------------------------------
    # Genel işlem #2: en yüksek öncelikli düğümü kaldır ve döndür
    # ------------------------------------------------------------------
    def extract_top(self) -> WordNode:
        """
        Şu anda kökte bulunan düğümü (çift anahtar kuralına göre önceliği
        en yüksek olan düğümü) heap'ten kaldırır ve döndürür.

        Heap boşalana kadar bu işlem tekrarlandığında, düğümler ödevin
        istediği sırayla gelir:
            ilk harfe göre A -> Z, ve ilk harfte eşitlik olduğunda
            en yüksek sayıdan en düşüğe.
        """
        if not self._data:
            raise IndexError("extract_top() boş bir heap üzerinde çağrıldı")

        top = self._data[0]
        last = self._data.pop()
        del self._index[top.word]

        if self._data:
            self._data[0] = last
            self._index[last.word] = 0
            self._sift_down(0)

        return top

    # ------------------------------------------------------------------
    # Hata ayıklama / görselleştirme yardımcıları
    # ------------------------------------------------------------------
    def print_array(self) -> None:
        """Heap'in alttaki dizisini yazdırır, örn. [Adana(2), Konya(2), ...]."""
        print("    array: [" + ", ".join(repr(n) for n in self._data) + "]")

    def print_tree(self) -> None:
        """
        Heap'i, kök en üstte ve çocukları dal bağlantılarıyla altına
        girintilenmiş şekilde ASCII bir ağaç olarak güzelce yazdırır.

        Herhangi bir terminal/konsol kodlamasında doğru görüntülenmesi
        için Unicode kutu çizim karakterleri yerine düz ASCII karakterleri
        ("+--", "`--", "|") kullanılır.
        """
        if not self._data:
            print("    (heap boş)")
            return
        print("    " + repr(self._data[0]))
        self._print_children(0, "    ")

    def _print_children(self, i: int, prefix: str) -> None:
        n = len(self._data)
        kids = [c for c in (2 * i + 1, 2 * i + 2) if c < n]
        for idx, child in enumerate(kids):
            is_last = idx == len(kids) - 1
            branch = "`-- " if is_last else "+-- "
            print(prefix + branch + repr(self._data[child]))
            extension = "    " if is_last else "|   "
            self._print_children(child, prefix + extension)


# ======================================================================
#  Metin işleme yardımcıları
# ======================================================================
def extract_words(text: str) -> list[str]:
    """
    Ham dosya içeriğini, göründükleri sırayla normalize edilmiş
    kelimelerden oluşan bir listeye dönüştürür.

    Normalizasyon kuralları:
      - Her şey küçük harfe çevrilir, böylece "Ankara", "ankara" ve
        "ANKARA" hepsi aynı kelime haline gelir.
      - Noktalama işaretleri, rakamlar, boşluklar ve harf olmayan diğer
        tüm karakterler ayraç (separator) olarak kabul edilir ve
        atılır: bir "kelime", art arda gelen Unicode harf
        karakterlerinin en uzun dizisi olarak tanımlanır. Örn.
        "Adana," / "(adana)" / "adana!" / "ADANA" hepsi "adana" olur;
        "word2vec" ise "word" ve "vec" şeklinde iki kelimeye dönüşür.
    """
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


# ======================================================================
#  Sürücü (driver): kelime akışından heap'i oluşturur, isteğe bağlı debug izi ile
# ======================================================================
def build_heap(words: list[str], debug: bool = False) -> DualKeyHeap:
    """
    Ödevin tarif ettiği şekilde ("heap, dosyadan okunan HER kelimeden
    sonra güncellenmelidir"), her kelimeyi tek tek yepyeni bir
    DualKeyHeap'e besler.

    ``debug`` True ise, her kelime işlendikten sonra heap'in dizi ve
    ağaç gösterimi yazdırılır; böylece adım adım davranış incelenebilir.
    """
    heap = DualKeyHeap()

    for step, word in enumerate(words, start=1):
        heap.insert_or_increment(word)

        if debug:
            print(f"Adım {step}: okunan kelime '{word}'")
            heap.print_array()
            heap.print_tree()
            print()

    return heap


def print_results(heap: DualKeyHeap) -> None:
    """
    Heap'i tekrar tekrar extract_top() çağrılarıyla boşaltır ve
    biçimlendirilmiş bir (kelime, sayı) tablosu yazdırır. Çift anahtar
    heap sıralaması sayesinde kelimeler zaten A->Z sırayla ve (ilk
    harfler eşitse) en yüksek sayıdan en düşüğe doğru gelir - ek bir
    sıralama adımına gerek yoktur.
    """
    if heap.is_empty():
        print("(kelime bulunamadı)")
        return

    print(f"{'Kelime':<20}{'Sayı':>6}")
    print("-" * 26)
    while not heap.is_empty():
        node = heap.extract_top()
        print(f"{node.word:<20}{node.count:>6}")


# ======================================================================
#  Demo modu: ödev açıklamasındaki çözülmüş örnek
# ======================================================================
DEMO_WORDS = ["Kocaeli", "Konya", "Van", "Ankara", "Konya", "Sivas", "Adana", "Adana"]


def run_demo() -> None:
    """
    Ödevde verilen, çözülmüş örneği tam olarak çalıştırır:

        Kocaeli, Konya, Van, Ankara, Konya, Sivas, Adana, Adana

    Her kelimeden sonra heap'in dizi + ağaç gösterimini, ardından nihai
    sıralanmış sonucu yazdırır.
    """
    print("=" * 60)
    print("DEMO: çift anahtarlı (dual-key) heap, adım adım")
    print("Girdi kelimeleri:", ", ".join(DEMO_WORDS))
    print("=" * 60)
    print()

    words = [w.lower() for w in DEMO_WORDS]
    heap = build_heap(words, debug=True)

    print("=" * 60)
    print("NİHAİ SONUÇ (heap'ten çıkarıldı, A->Z, sayı azalan)")
    print("=" * 60)
    print_results(heap)


# ======================================================================
#  Ana giriş noktası
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bir TXT dosyasındaki kelime frekanslarını sayar ve "
            "A->Z (ilk harfe göre), aynı ilk harfe sahip kelimeler için "
            "frekansa göre azalan sırada yazdırır. Sıralama tamamen "
            "özel bir çift anahtarlı (dual-key) heap tarafından yapılır."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="girdi .txt dosyasının yolu",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="her kelimeden sonra heap'in dizi ve ağaç durumunu yazdır",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="dosya okumak yerine ödevdeki hazır Kocaeli/Konya/Van/... "
        "örneğini çalıştır",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    filepath = args.file
    if not filepath:
        filepath = input("Bir .txt dosyasının yolunu girin: ").strip()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"Hata: '{filepath}' açılamadı: {exc}")
        sys.exit(1)

    words = extract_words(text)

    if not words:
        print("Dosyada hiç kelime bulunamadı.")
        return

    heap = build_heap(words, debug=args.debug)

    if args.debug:
        print("=" * 60)
        print("NİHAİ SONUÇ")
        print("=" * 60)

    print_results(heap)


if __name__ == "__main__":
    main()
