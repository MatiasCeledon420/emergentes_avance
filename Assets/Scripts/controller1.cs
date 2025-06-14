using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class Controller2 : MonoBehaviour {
    public GameObject  game;
    public TextMeshProUGUI gameHitCountOut;
    public TextMeshProUGUI letterOut;
    public Image imageOut;

    public GameObject results;
    public TextMeshProUGUI resultsHitCountOut;

    private int hitCount = 0; 
    private bool showingResults = false;

    private Sign targetSign;

    // Referencia para comunicarse con el script LetterReceiver
    private LetterReceiver receiver;
    private List<Sign> signs = new List<Sign>();

    public GameObject goalUI;
    public AnimationCurve barCurve;

    public bool Should_End() {
        return ( hitCount >= State.data.goal );
    }

    public string Get_Input_Text() {
        return receiver.Get_Letter();
        // return ( !string.IsNullOrEmpty(Input.inputString) ) ? Input.inputString : null;
    }

    public void Target_Sign_Set() {
        Debug.Assert(signs.Count != 0);

        if ( State.data.randomSelection ) {
            targetSign = signs[Random.Range(0, signs.Count)];
        } else {
            targetSign = signs[hitCount % signs.Count];
        }

        imageOut.sprite = targetSign.sprite;
        letterOut.text  = targetSign.text;
    }

    private void Update_Goal_Bar() {
        Image fill = goalUI.transform.Find("Fill").gameObject.GetComponent<Image>();
        TextMeshProUGUI text = goalUI.transform.Find("Text").gameObject.GetComponent<TextMeshProUGUI>();

        if ( State.data.goal != int.MaxValue ) {
            StartCoroutine(Bar_Fill_Animation(fill, fill.fillAmount, (float)hitCount / (float)State.data.goal, 0.2f));
            text.text = $"{hitCount}/{State.data.goal}";
        } else {
            fill.fillAmount = 1f;
            text.text = $"{hitCount}";
        }
    }

    private IEnumerator Bar_Fill_Animation(Image bar, float from, float to, float duration) {
        float elapsed = 0f;

        while (elapsed < duration) {
            elapsed += Time.deltaTime;
            float t = barCurve.Evaluate(Mathf.Clamp01(elapsed / duration));
            bar.fillAmount = Mathf.Lerp(from, to, t);
            yield return null;
        }

        bar.fillAmount = to;
    }

    public void Init() {
        showingResults = false;

        game.SetActive(true);
        results.SetActive(false);

        if ( State.data.time != int.MaxValue ) {
            GetComponent<Timer>().Set(State.data.time, 0);
        }

        hitCount = 0;
        goalUI.transform.Find("Fill").gameObject.GetComponent<Image>().fillAmount = 0;
        Target_Sign_Set();
        Update_Goal_Bar();
    }

    void Start() {
        // Al iniciar, busca el componente LetterReceiver en la escena
        receiver = FindObjectOfType<LetterReceiver>();
        signs = State.data.signs.Where(iter => iter.enable).ToList(); // Find enabled signs
        Init();
    }

    void Update() {
        if ( showingResults ) { return; }

        string input = Get_Input_Text(); // Obtener input del usuario

        if ( input != null ) {
            if ( targetSign.text.ToString().ToLower() == input.ToLower() ) { // Compara la letra recibida con la letra objetivo
                hitCount++;
                Update_Goal_Bar();
                
                // Comprueba si aún quedan señas por mostrar
                if ( !Should_End() ) {
                    Target_Sign_Set();
                } else {
                    ShowResults();
                }
            }
        }
    }

    public void ShowResults() {
        showingResults = true;

        game.SetActive(false);
        results.SetActive(true);

        resultsHitCountOut.text = $"Aciertos: {hitCount}";
    }
}