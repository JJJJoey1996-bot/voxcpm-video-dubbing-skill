VOICE_PERSONA_PRESETS = [
    {
        "name": "无",
        "category": "自定义",
        "control": "",
        "text": "",
    },
    {
        "name": "温柔女友",
        "category": "情感陪伴",
        "control": "年轻女性，温柔、亲近、带一点微笑感，语速自然。",
        "text": "今天辛苦了，先深呼吸一下。接下来的时间，我们慢慢来。",
    },
    {
        "name": "治愈电台",
        "category": "播客旁白",
        "control": "成熟知性的女声，平稳、柔和、像深夜电台主播。",
        "text": "欢迎来到今晚的节目。把注意力放回呼吸，我们一起让心慢下来。",
    },
    {
        "name": "播音口号",
        "category": "官方示例",
        "control": "热情洋溢的中年男性播音员，声音较为低沉，富有磁性与感染力，带着逐渐密集的节奏感呼喊宣讲口号",
        "text": "全力以赴，乘势而上，向着目标坚定前进。",
    },
    {
        "name": "英语讲解员",
        "category": "Multilingual",
        "control": "Professional English female voice, calm, articulate, medium pace, suitable for tutorials.",
        "text": "Welcome back. In this section, we will walk through the workflow step by step.",
    },
    {
        "name": "粤语大叔",
        "category": "方言",
        "control": "38岁广东男性，街坊感强，声音厚实自然，语气爽快利落，像茶餐厅里熟练招呼客人的本地大叔。",
        "text": "伙計，唔該一個A餐，凍奶茶少甜！",
    },
    {
        "name": "四川话幺儿",
        "category": "方言",
        "control": "24岁四川女生，语气灵动直爽，带一点俏皮和烟火气，像成都街头跟朋友摆龙门阵的年轻姑娘。",
        "text": "幺儿，哈戳戳得你屋头来噶！",
    },
    {
        "name": "吴语阿姨",
        "category": "方言",
        "control": "46岁江浙女性，说话细腻温和，语速从容，声音柔软带一点生活化的亲切感，像社区里耐心招呼人的本地阿姨。",
        "text": "侬今朝吃过了伐？要末一道去汏点小菜回来。",
    },
    {
        "name": "东北话唠嗑",
        "category": "方言",
        "control": "33岁东北男性，嗓音敞亮，语气热情外放，节奏明快，像和熟人站在门口唠嗑的爽朗老铁。",
        "text": "你搁这整啥玩意儿呢？",
    },
    {
        "name": "河南话大叔",
        "category": "方言",
        "control": "45岁河南男性，朴实直接，声音偏厚，语气自然接地气，像村口熟络聊天的本地叔叔。",
        "text": "恁这是弄啥嘞？晌午吃啥饭？",
    },
    {
        "name": "陕西话伙计",
        "category": "方言",
        "control": "36岁陕西男性，语气豪爽，发声结实有劲，带一点西北口音的干脆劲儿，像馆子里热情招呼人的本地伙计。",
        "text": "走撒，咥面去！今儿这碗油泼面香得很。",
    },
    {
        "name": "山东话大姨",
        "category": "方言",
        "control": "52岁山东女性，声音明亮有劲，语气热心敞亮，像家里做饭时一边招呼晚辈一边念叨的本地大姨。",
        "text": "快来吃饭咧，锅里刚出锅的饺子可鲜亮了。",
    },
    {
        "name": "天津话大哥",
        "category": "方言",
        "control": "34岁天津男性，说话带点松弛幽默感，节奏轻快，像胡同口随口接话、爱逗乐的本地大哥。",
        "text": "嘛呢您呐？甭着急，咱这就给您办利索喽。",
    },
    {
        "name": "闽南语阿伯",
        "category": "方言",
        "control": "58岁闽南男性，声音略低沉，语气稳当亲切，像老街巷口慢慢讲话、很有生活阅历的本地阿伯。",
        "text": "恁今仔日欲去叨位？食饱未，莫着急啦。",
    },
]


VOICE_TAG_PRESETS = [
    {"label": "笑", "value": "[laughing]"},
    {"label": "叹气", "value": "[sigh]"},
    {"label": "迟疑", "value": "[Uhm]"},
    {"label": "安静", "value": "[Shh]"},
    {"label": "疑问-ah", "value": "[Question-ah]"},
    {"label": "疑问-ei", "value": "[Question-ei]"},
    {"label": "疑问-en", "value": "[Question-en]"},
    {"label": "疑问-oh", "value": "[Question-oh]"},
    {"label": "惊讶-wa", "value": "[Surprise-wa]"},
    {"label": "惊讶-yo", "value": "[Surprise-yo]"},
    {"label": "不满-hnn", "value": "[Dissatisfaction-hnn]"},
]


COOKBOOK_GUIDE_HTML = """
<div class="tips">
<strong>官方填写建议</strong>
<br>1. 多语种场景：多数情况下直接写目标语言正文即可，不必额外加语言标签。
<br>2. 中文方言：正文尽量直接写成地道方言表达，效果通常明显好于普通话硬套。
<br>3. 方言控制：Control Instruction 里尽量只写方言名，例如 <code>Cantonese</code>，不要叠太多复杂音色描述。
<br>4. 非语言标签：只用官方推荐标签，且数量要少，放在正文里点到为止。
<br>5. 声音设计：把身份、声音质感、表达状态写成一条清晰完整的控制指令。
<br>6. 声音克隆：尽量使用 5 秒以上、干净稳定的参考音频；上传后仍可继续加控制指令微调情绪和语速。
</div>
"""
