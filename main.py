from app_sandbox import StudentDiagnostic, assess_student_state
from citation_agent import citation_validator
from integrity_agent import check_integrity
from literature_agent import literature_search
from methodology_agent import methodology_mentor
from socratic_agent import socratic_refiner
from writing_agent import writing_scaffolder


def main():
    print("--- 🎓 My Research Assistant ---")
    print("Type 'quit' or 'exit' at any time to leave.\n")

    while True:
        user_input = input("Describe where you are in your research:\n> ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! Best of luck with your research. 👋")
            break

        if not user_input:
            continue

        diagnosis_raw = assess_student_state(user_input)
        diagnosis = StudentDiagnostic.model_validate_json(diagnosis_raw)

        print(f"\nStage: {diagnosis.current_stage}")
        print(f"\n{diagnosis.initial_response}\n")

        stage = diagnosis.current_stage.strip()

        if stage == "1_Ideation":
            print("--- 🧠 Socratic Mentor ---\n")
            mentor_response = socratic_refiner(user_input)
            print(mentor_response.content)
        elif stage == "2_Literature":
            print("--- 📚 Literature Researcher ---\n")
            search_choice = input(
                "Would you like a 'full' or 'recent' literature search?\n> "
            ).strip().lower()
            if search_choice == "full":
                search_type = "comprehensive"
            elif search_choice == "recent":
                search_type = "recent"
            else:
                print("Invalid choice. Defaulting to 'recent' search.\n")
                search_type = "recent"
            literature_results = literature_search(user_input, search_type)
            print(literature_results)
        elif stage == "3_Methodology":
            print("--- 📊 Methodology Coach ---\n")
            methodology_response = methodology_mentor(user_input)
            print(methodology_response.content)
        elif stage == "4_Writing":
            print("--- ✍️ Writing Scaffolder ---\n")
            writing_response = writing_scaffolder(user_input)
            print(writing_response.content)
        elif stage == "5_Review":
            print("--- 📎 Citation Validator ---\n")
            ref_style = input(
                "What referencing style do you need? (e.g., APA, Harvard)\n> "
            ).strip()
            if not ref_style:
                ref_style = "APA"
            citation_response = citation_validator(user_input, referencing_style=ref_style)
            print(citation_response.content)
        elif stage == "6_Integrity":
            print("--- 🛡️ Integrity Agent ---\n")
            integrity_results = check_integrity(user_input)
            print(integrity_results)
        else:
            print(f"Unrecognized stage '{diagnosis.current_stage}'.")
            print(f"The recommended agent ({diagnosis.recommended_agent}) is currently under construction.")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
