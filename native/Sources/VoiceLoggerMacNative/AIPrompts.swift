import Foundation

/// Prompts of the native AI engine: a word-for-word port of `PROMPTS` in
/// `summarizer.py`, so both engines ask the model exactly the same.
struct AIPrompts: Sendable {
    let role: String
    let job: String
    let jobOpen: String
    let minutes: String
    let sections: String
    let transcript: String
    /// `{index}` and `{chunk}` are replaced.
    let chunk: String
    let part: String
    let consolidated: String
    /// `{previous}` and `{new}` are replaced.
    let update: String
    /// `{today}` is replaced.
    let tasks: String
    /// `{limit}` is replaced.
    let questions: String
    let known: String
    let latest: String
    let trimmed: String

    // "ask" labels
    let notes: String
    let memory: String
    let recent: String
    let meetingTranscript: String
    let question: String
    let web: String
    let reliable: String
    let openInstruction: String
    let openInstructionNoWeb: String
    let given: String
    let complement: String
    let previous: String
    let extend: String
    let instructionShort: String
    let instructionLong: String

    // Lengths (Ideas answers and web complements)
    let openLong: String
    let openShort: String
    let complementLong: String
    let complementShort: String

    static func forLanguage(_ language: String) -> AIPrompts {
        language == "es" ? .es : .en
    }

    static let en = AIPrompts(
        role: "You are a meeting assistant.",
        job: "Your job is to help me follow a meeting from its live transcript. "
            + "Always answer in English, even if the transcript is in another language. "
            + "Use only what the transcript says; do not invent names, dates or numbers. "
            + "Reply only with the final result: do not show your reasoning or repeat these instructions.",
        jobOpen: "Your job is to help me during a brainstorming meeting. "
            + "Always answer in English, even if the transcript is in another language. "
            + "Use the meeting as context and add your own knowledge and ideas, but never present them "
            + "as something said in the meeting, and do not invent facts about the meeting. "
            + "Reply only with the final result: do not show your reasoning or repeat these instructions.",
        minutes: "Write short meeting notes in Markdown from the transcript below, "
            + "using exactly these sections:\n",
        sections: "## Summary\n(2-3 sentences)\n"
            + "## Key topics\n(bullets)\n"
            + "## Key questions\n(bullets: the most important questions asked or raised, at most 5; "
            + "if someone asked it, copy it word for word)\n"
            + "## Decisions\n(bullets)\n"
            + "## Action items\n(bullets, with owner if mentioned)\n",
        transcript: "Transcript:",
        chunk: "Extract the key ideas, questions, decisions and action items (with owner "
            + "if mentioned) from this part of the meeting, as short bullets.\n\nPart {index}:\n{chunk}",
        part: "Part",
        consolidated: "(Consolidated notes from the whole meeting)",
        update: "These are the current notes of a meeting in progress:\n\n{previous}\n\n"
            + "This is the new part of the transcript:\n{new}\n\n"
            + "Update the notes so they cover the whole meeting so far: keep the important "
            + "earlier points, add the new ones and drop questions that were already answered. "
            + "Use exactly these sections:\n",
        tasks: "From the meeting notes and transcript below, list the action items: concrete things "
            + "someone has to do after the meeting. Write one per line in exactly this format:\n"
            + "- TASK | OWNER | DUE\n"
            + "TASK: short imperative sentence. OWNER: who is responsible, or - if not mentioned. "
            + "DUE: the deadline as YYYY-MM-DD if one was mentioned (today is {today}), or -.\n"
            + "Write at most 10 lines and nothing else. If there are no action items, write: - none",
        questions: "From the latest part of the meeting below, list the important questions that were asked or "
            + "raised. If someone asked it, copy it word for word. Do not repeat the questions already "
            + "listed, even reworded. One per line, starting with '- ', at most {limit}, and nothing else. "
            + "If there are no new questions, write: - none",
        known: "Questions already listed:",
        latest: "Latest part of the meeting:",
        trimmed: "(… transcript trimmed: only the most recent part …)",
        notes: "Notes of the meeting so far:",
        memory: "Earlier parts of the meeting related to the question:",
        recent: "Most recent part of the transcript:",
        meetingTranscript: "Meeting transcript:",
        question: "Question:",
        web: "Web search results:",
        reliable: "reliable source",
        openInstruction: "Answer as an expert who explains things clearly, so the answer can be shared as it is. Adapt the "
            + "format to the question: for how-to questions, give numbered step-by-step instructions; for 'which is "
            + "best' or 'what should I choose' questions, give a clear recommendation first and then explain why, "
            + "comparing it with the alternatives; for other questions, explain the reasoning in well-written "
            + "paragraphs with concrete examples. Explain the why behind each point and do not use labels such as "
            + "'Key insight' or 'Next step'. Use the meeting as context only where it is relevant (say so briefly), "
            + "add your own knowledge, and cite facts from the web results with their number, like [1] or [2]. If "
            + "sources disagree, trust the ones marked (reliable source).",
        openInstructionNoWeb: "Answer as an expert who explains things clearly, so the answer can be shared as it is. Adapt the "
            + "format to the question: for how-to questions, give numbered step-by-step instructions; for 'which is "
            + "best' or 'what should I choose' questions, give a clear recommendation first and then explain why, "
            + "comparing it with the alternatives; for other questions, explain the reasoning in well-written "
            + "paragraphs with concrete examples. Explain the why behind each point and do not use labels such as "
            + "'Key insight' or 'Next step'. Use the meeting as context only where it is relevant (say so briefly) "
            + "and add your own knowledge.",
        given: "Answer already given:",
        complement: "Using only the web results, add what they contribute to the answer already given: new facts, "
            + "figures, examples or corrections, explained in the same style (steps or explanatory paragraphs, no "
            + "labels). Do not repeat the answer already given. Cite web facts with their number, like [1] or [2]. "
            + "If sources disagree, trust the ones marked (reliable source). If the web results add nothing useful, "
            + "reply only: NOTHING NEW",
        previous: "Answer based on the meeting:",
        extend: "Extend the answer with useful information from the web results. Keep what the "
            + "meeting said, clearly separate it from what comes from the web, and cite web facts "
            + "with their number, like [1] or [2]. If sources disagree, trust the ones marked "
            + "(reliable source). If the web results are not relevant, say so.",
        instructionShort: "Answer concisely (2-4 sentences) using only this information. If the answer "
            + "is not there, say so clearly instead of making it up.",
        instructionLong: "Give a thorough, well-organized answer using only this information: explain the "
            + "context, what was said (and by whom, if known), any decisions, numbers or dates, "
            + "and the open points or next steps. Use short paragraphs or bullet points "
            + "(about 150-300 words). If the answer is not there, say so clearly instead of "
            + "making it up.",
        openLong: "Be as complete as the question needs (about 250-500 words).",
        openShort: "Keep it short: 3-5 sentences, or at most 5 steps.",
        complementLong: "About 150-250 words: build more elaborate ideas, examples or steps with what the web adds, "
            + "in short paragraphs or bullets.",
        complementShort: "2-4 sentences."
    )

    static let es = AIPrompts(
        role: "Eres un asistente de reuniones.",
        job: "Tu trabajo es ayudarme a seguir una reunión a partir de su transcripción en vivo. "
            + "Responde SIEMPRE en español. Usa solo lo que dice la transcripción; "
            + "no inventes nombres, fechas ni cifras. "
            + "Responde solo con el resultado final: no muestres tu razonamiento ni repitas estas instrucciones.",
        jobOpen: "Tu trabajo es ayudarme durante una reunión de lluvia de ideas. "
            + "Responde SIEMPRE en español. Usa la reunión como contexto y añade tu propio conocimiento e "
            + "ideas, pero nunca los presentes como algo dicho en la reunión ni inventes datos de la reunión. "
            + "Responde solo con el resultado final: no muestres tu razonamiento ni repitas estas instrucciones.",
        minutes: "A partir de la siguiente transcripción de reunión, genera un acta breve "
            + "en formato Markdown con estas secciones exactas:\n",
        sections: "## Resumen\n(2-3 frases)\n"
            + "## Temas tratados\n(viñetas)\n"
            + "## Preguntas clave\n(viñetas: las preguntas más importantes que se han hecho o planteado, máximo 5; "
            + "si alguien la hizo, cópiala tal cual)\n"
            + "## Decisiones\n(viñetas)\n"
            + "## Tareas / pendientes\n(viñetas, con responsable si se menciona)\n",
        transcript: "Transcripción:",
        chunk: "Extrae las ideas clave, preguntas, decisiones y tareas (con responsable si se "
            + "menciona) de este fragmento de reunión, en viñetas cortas.\n\nFragmento {index}:\n{chunk}",
        part: "Parte",
        consolidated: "(Notas consolidadas de toda la reunión)",
        update: "Estas son las notas actuales de una reunión en curso:\n\n{previous}\n\n"
            + "Esta es la parte nueva de la transcripción:\n{new}\n\n"
            + "Actualiza las notas para que cubran toda la reunión hasta ahora: conserva lo "
            + "importante de antes, añade lo nuevo y quita las preguntas que ya se respondieron. "
            + "Usa exactamente estas secciones:\n",
        tasks: "A partir de las notas y la transcripción de abajo, lista las tareas: cosas concretas "
            + "que alguien tiene que hacer después de la reunión. Escribe una por línea con este "
            + "formato exacto:\n"
            + "- TAREA | RESPONSABLE | FECHA\n"
            + "TAREA: frase corta en infinitivo. RESPONSABLE: quién la hace, o - si no se dijo. "
            + "FECHA: la fecha límite como AAAA-MM-DD si se mencionó (hoy es {today}), o -.\n"
            + "Escribe como máximo 10 líneas y nada más. Si no hay tareas, escribe: - ninguna",
        questions: "De la parte más reciente de la reunión, abajo, lista las preguntas importantes que se hicieron "
            + "o plantearon. Si alguien la hizo, cópiala tal cual. No repitas las preguntas ya listadas, ni "
            + "con otras palabras. Una por línea, empezando con '- ', como máximo {limit}, y nada más. "
            + "Si no hay preguntas nuevas, escribe: - ninguna",
        known: "Preguntas ya listadas:",
        latest: "Parte más reciente de la reunión:",
        trimmed: "(… transcripción recortada: solo la parte más reciente …)",
        notes: "Notas de la reunión hasta ahora:",
        memory: "Partes anteriores de la reunión relacionadas con la pregunta:",
        recent: "Parte más reciente de la transcripción:",
        meetingTranscript: "Transcripción de la reunión:",
        question: "Pregunta:",
        web: "Resultados de la búsqueda en internet:",
        reliable: "fuente confiable",
        openInstruction: "Responde como un experto que explica con claridad, para que la respuesta se pueda compartir tal "
            + "cual. Adapta el formato a la pregunta: si es de cómo hacer algo, da instrucciones paso a paso "
            + "numeradas; si es de cuál es mejor o qué elegir, da primero una recomendación clara y luego explica "
            + "por qué, comparándola con las alternativas; en las demás, explica el razonamiento en párrafos bien "
            + "escritos con ejemplos concretos. Explica el porqué de cada punto y no uses etiquetas como «Idea "
            + "clave» o «Próximo paso». Usa la reunión como contexto solo donde sea relevante (dilo brevemente), "
            + "añade tu propio conocimiento y cita los datos de internet con su número, como [1] o [2]. Si las "
            + "fuentes no coinciden, confía en las marcadas (fuente confiable).",
        openInstructionNoWeb: "Responde como un experto que explica con claridad, para que la respuesta se pueda compartir tal "
            + "cual. Adapta el formato a la pregunta: si es de cómo hacer algo, da instrucciones paso a paso "
            + "numeradas; si es de cuál es mejor o qué elegir, da primero una recomendación clara y luego explica "
            + "por qué, comparándola con las alternativas; en las demás, explica el razonamiento en párrafos bien "
            + "escritos con ejemplos concretos. Explica el porqué de cada punto y no uses etiquetas como «Idea "
            + "clave» o «Próximo paso». Usa la reunión como contexto solo donde sea relevante (dilo brevemente) y "
            + "añade tu propio conocimiento.",
        given: "Respuesta ya dada:",
        complement: "Usando solo los resultados de internet, agrega lo que aportan a la respuesta ya dada: datos nuevos, "
            + "cifras, ejemplos o correcciones, explicados con el mismo estilo (pasos o párrafos explicativos, sin "
            + "etiquetas). No repitas la respuesta ya dada. Cita los datos de internet con su número, como [1] o "
            + "[2]. Si las fuentes no coinciden, confía en las marcadas (fuente confiable). Si los resultados no "
            + "aportan nada útil, responde solo: NADA NUEVO",
        previous: "Respuesta basada en la reunión:",
        extend: "Amplía la respuesta con información útil de los resultados de internet. Conserva lo "
            + "que dijo la reunión, sepáralo claramente de lo que viene de internet y cita los datos "
            + "de internet con su número, como [1] o [2]. Si las fuentes no coinciden, confía en las "
            + "marcadas (fuente confiable). Si los resultados no son relevantes, dilo.",
        instructionShort: "Responde de forma concisa (2-4 frases) usando solo esta información. Si la "
            + "respuesta no aparece, dilo claramente en vez de inventarla.",
        instructionLong: "Da una respuesta completa y bien organizada usando solo esta información: explica "
            + "el contexto, qué se dijo (y quién, si se sabe), las decisiones, cifras o fechas, y "
            + "los puntos abiertos o próximos pasos. Usa párrafos cortos o viñetas (unas 150-300 "
            + "palabras). Si la respuesta no aparece, dilo claramente en vez de inventarla.",
        openLong: "Sé tan completo como la pregunta lo necesite (unas 250-500 palabras).",
        openShort: "Sé breve: 3-5 frases, o como mucho 5 pasos.",
        complementLong: "Unas 150-250 palabras: arma ideas más elaboradas, ejemplos o pasos con lo que aporta internet, "
            + "en párrafos cortos o viñetas.",
        complementShort: "2-4 frases."
    )
}
