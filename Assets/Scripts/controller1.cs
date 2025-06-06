using System.Collections;
using System.Collections.Generic;
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

    public void ChooseSign() {
        targetSign = State.data.signs[hitCount];
        imageOut.sprite = targetSign.sprite;
        letterOut.text  = targetSign.text.ToString();
    }

    void Start() {
        ChooseSign();
    }

    void Update() {
        if ( showingResults ) { return; }

        if ( Input.anyKeyDown ) {
            if ( Input.inputString.Length != 1 ) { return; }

            char ch = Input.inputString[0];

            if ( targetSign.text.ToLower() == ch.ToString() ) {
                hitCount++;
                if ( hitCount < State.data.signs.Length ) {
                    ChooseSign();
                } else {
                    ShowResults();
                }
            }
        }    

        gameHitCountOut.text = hitCount.ToString();
    }

    public void Restart() {
        showingResults = false;

        game.SetActive(true);
        results.SetActive(false);

        hitCount = 0;
        ChooseSign();
    }

    public void ShowResults() {
        showingResults = true;

        game.SetActive(false);
        results.SetActive(true);

        resultsHitCountOut.text = hitCount.ToString();
    }
}