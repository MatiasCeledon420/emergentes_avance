using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class controller : MonoBehaviour {
    public GameObject  game;
    public TextMeshProUGUI gameHitCountOut;
    public TextMeshProUGUI letterOut;
    public Image imageOut;

    public GameObject results;
    public TextMeshProUGUI resultsHitCountOut;

    private int hitCount = 0; 
    private bool showingResults = false;

    private Sign targetSign;
    
    // --- AÑADIDO ---
    // Referencia al componente LetterReceiver
    private LetterReceiver receiver;

    public void ChooseHand() {
        targetSign = State.data.signs[Random.Range(0, State.data.signs.Length)].Clone();
        imageOut.sprite = targetSign.sprite;
        letterOut.text  = targetSign.text.ToString();
    }

    void Start() {
        // --- AÑADIDO ---
        // Busca el objeto LetterReceiver en la escena
        receiver = FindObjectOfType<LetterReceiver>(); 
        ChooseHand();
    }

    // --- MÉTODO MODIFICADO ---
    // Se reemplaza la lógica de entrada del teclado por la del LetterReceiver
    void Update() {
        if ( showingResults ) { return; }
        
        // Obtener la letra del LetterReceiver
        string receivedLetter = (receiver != null && !string.IsNullOrEmpty(receiver.Letter)) ? receiver.Letter : " ";

        // Si se recibió una letra válida (no es el valor por defecto " ")
        if ( receivedLetter != " " ) {
            // Compara la letra recibida (ya en mayúsculas) con la letra del signo objetivo
            if ( targetSign.text.ToString().ToUpper() == receivedLetter ) {
                hitCount++;
                ChooseHand();
            }
            
            // Limpia la letra en el receiver para no procesarla de nuevo
            receiver.ClearLetter();
        }

        gameHitCountOut.text = hitCount.ToString();
    }

    public void Restart() {
        showingResults = false;
        game.SetActive(true);
        results.SetActive(false);
        hitCount = 0;
        ChooseHand(); // Elige una nueva mano al reiniciar
    }

    public void ShowResults() {
        showingResults = true;
        game.SetActive(false);
        results.SetActive(true);
        resultsHitCountOut.text = hitCount.ToString();
    }
}