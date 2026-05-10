"""对比两条已带前缀的文本在 GGUF 下的向量余弦相似度（与检索链路一致）。

``Query: …``（问句检索）或 ``Document: …``（文档/chunk）均可：须把**完整前缀+正文**连成一条，
再交给 ``embed_prefixed``。（勿对已含 ``Query: `` 的串再调用 ``embed_query``，否则会双前缀。）

用法（项目根、已配置 .env）::

    uv run python scripts/compare_gguf_embed_pair.py

    uv run python scripts/compare_gguf_embed_pair.py \\
      --text1 "Query: 你好" \\
      --text2 "Query: 你好？"
"""

from __future__ import annotations

import argparse
import math
import sys

from dotenv import load_dotenv

from iskra_engine.embeddings.gguf_embed import embed_prefixed


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两向量均已 L2 归一时等于点积。"""
    if len(a) != len(b):
        msg = f"维数不一致: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def l2_distance_unit(a: list[float], b: list[float]) -> float:
    """单位向量之间的欧氏距离：sqrt(2 - 2 cosθ)。"""
    c = cosine_similarity(a, b)
    return math.sqrt(max(0.0, 2.0 - 2.0 * c))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "对比两条已编码字符串（通常含 Query: 或 Document: 前缀）的 "
            "embedding 余弦相似度（gguf_embed，与线上标尺一致）。"
        ),
    )
    parser.add_argument(
        "--text1",
        default=(
            "Query: 毛泽东是不是建国后接见过美国作家记者埃德加·斯诺（Edgar Snow）很多次？他们说了什么"
            # "Document: - ８８\n\n  威·罗雪尔《德国国民经济学史》１８７４年慕尼黑版第１０２１—１０２２页 （Ｗ．Ｒｏｓｃｈｅｒ．《Ｇｅｓｃｈｉｃｈｔｅ ｄｅｒ Ｎａｔｉｏｎａｌ－Ｏｅｋｏｎｏｍｉｅ ｉｎ Ｄｅｕｔｓｃｈ －ｌａｎｄ》．Ｍüｎｃｈｅｎ，１８７４，Ｓ．１０２１—１０２２）。罗雪尔在这里评述了马克思的经济学理论。—— 第９４页。\n\n- ８９\n\n  指１８８８年９月２０日《纽约人民报》发表的同恩格斯的谈话（见《马克思恩格斯全集》中文版第２１卷第５７１—５７２页）。恩格斯在美国旅行期间不愿意会见德国各社会主义组织的代表，因此竭力回避与新闻界代表谈话。《纽约人民报》编辑约纳斯获悉恩格斯留在纽约之后，就派以前第一国际的活动家泰·库诺作为他的代表走访恩格斯，恩格斯同他进行了谈话。谈话内容事先未经恩格斯的同意就发表出去了。—— 第 ９７、３７６页。\n\n- ９０\n\n  恩格斯显然是指费边社社员悉尼·维伯和比阿特里萨·维伯、乔治· 肖伯纳、爱德华·皮斯（见注１７２）。—— 第１０４页。\n\n- ９１\n\n  恩格斯曾经打算写些关于旅美的旅途特写，正如《美国旅行印象》片断 （见《马克思恩格斯全集》中文版第２１卷第５３４—５３６页）和保存下来的若干札记所证明的，他本想在特写中描述一下美国的社会政治生活。 但是他的这个想法未能实现。—— 第１０９页。\n\n- ９２\n\n  倍倍尔打算写一本关于魏特林的长篇著作，在这一著作中他还想探讨关于“四十年代的社会运动”问题。他请恩格斯帮助他搜集材料。—— 第１０９页。\n\n- ９３"
        ),
        help="第一段完整串（默认缺句末「？」的 Query 示例）；可换成 Document: 等",
    )
    parser.add_argument(
        "--text2",
        default=(
            "Query: 毛泽东是不是建国后接见过美国作家记者埃德加·斯诺（Edgar Snow）很多次？他们说了什么？"
            # "Document: # 与斯诺的谈话——关于文化大革命[^1]\n\n*（一九七〇年十二月十八日）*\n\n斯诺：我经常想给你写信，但我真正写信打扰你还只有这一次。\n\n毛泽东：怎么是打扰呢？上次，一九六五年，我就叫你找我嘛。你早找到我，骂人，我就早让你来看中国的文化大革命，看全面内战，all-round civil war，我也学了这句话了。到处打，分两派，每一个工厂分两派，每一个学校分两派，每一个省分两派，每一个县分两派，每一个部也是这样，外交部就是两派。你不搞这个东西也不行，一是有反革命，二是有走资派。外交部就闹得一塌糊涂。有一个半月失去了掌握，这个权掌握在反革命手里。\n\n斯诺：是不是火烧英国代办处[^2]的时候？\n\n毛泽东：就是那个时期。一九六七年七月 July 和八月 August 两个月不行了，天下大乱了。这一来就好了，他就暴露了，不然谁知道啊？！多数还是好的，有少数人是坏人。这个敌人叫“五·一六”[^3]。\n\n斯诺：有一个问题我还不大清楚，即主席对我讲这些，是供公开发表用，还是作为介绍背景材料，还是朋友之间的交谈，还是三者兼而有之。\n\n毛泽东：不供发表。就是作为学者，研究者，研究社会情况，研究将来，研究历史嘛。我看你发表跟周恩来总理的谈话比较好，同我的不要发表。意大利杂志上的这一篇[^4]我看了，我是看从外国文翻译成中文的。\n\n斯诺：你看写得可以不可以？\n\n毛泽东：可以嘛。你的那些什么错误有什么要紧？比如，说我是个人崇拜。你们美国人才是个人崇拜多呢！你们的国都就叫作华盛顿。你们的华盛顿所在的那个地方就叫作哥伦比亚区。\n\n斯诺：每个州里面还起码都有一个名为华盛顿的市镇。\n\n毛泽东：可讨嫌了！科学上的发明我赞成，比如，达尔文、康德[^5]，甚至还有你们美国的科学家，主要是那个研究原始社会的摩根[^6]，他的书马克思、恩格斯都非常欢迎。从此才知道有原始社会。总要有人崇拜嘛！你斯诺没有人崇拜你，你就高兴啦？你的文章、你的书写出来没有人读你就高兴啦？总要有点个人崇拜，你也有嘛。你们美国每个州长、每个总统、每个部长没有一批人崇拜他怎么混得下去呢！"
        ),
        help="第二段完整串（默认句末多「？」的 Query 示例）；可换成 Document: 等",
    )
    args = parser.parse_args()

    load_dotenv()
    text1 = args.text1.strip()
    text2 = args.text2.strip()
    if not text1 or not text2:
        print("text1/text2 不能为空", file=sys.stderr)
        return 1

    try:
        vec1 = embed_prefixed(text1)
        vec2 = embed_prefixed(text2)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1

    cos = cosine_similarity(vec1, vec2)
    l2 = l2_distance_unit(vec1, vec2)

    print(f"dim={len(vec1)}")
    print(f"cosine_similarity={cos:.6f}")
    print(f"l2_distance (unit vectors)={l2:.6f}")
    print()
    print("--- text1 ---")
    print(repr(text1))
    print("--- text2 ---")
    print(repr(text2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
