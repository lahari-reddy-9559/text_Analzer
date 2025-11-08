    elif choice == "keywords":
        st.subheader("🧩 Key Keywords")
        kw = extract_keywords(text_input)
        st.success(", ".join(kw))

    elif choice == "topics":
        st.subheader("📊 Extracted Topics")
        topics = extract_topics(text_input)
        for i, t in enumerate(topics, 1):
            st.info(f"Topic {i}: {t}")

    elif choice == "insights":
        st.subheader("🎯 Actionable Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        kw = extract_keywords(text_input)
        topics = extract_topics(text_input)
        recs = generate_recommendations(text_input, top_sent, kw, topics)
        for r in recs:
            st.markdown(f"- {r}")
