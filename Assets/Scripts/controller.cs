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

    public void ChooseHand() {
        // if ( targetSign != null ) { Destroy(targetSign.hand); }
        targetSign = State.data.signs[Random.Range(0, State.data.signs.Length)].Clone();
        // , new Vector3(0, 0, 0)
        // targetSign.hand = Instantiate(targetSign.hand, game.transform);
        imageOut.sprite = targetSign.sprite;
        letterOut.text  = targetSign.text.ToString();
    }

    void Start() {
        ChooseHand();
    }

    void Update() {
        if ( showingResults ) { return; }

        if ( Input.anyKeyDown ) {
            if ( Input.inputString.Length != 1 ) { return; }

            char ch = Input.inputString[0];

            if ( targetSign.text.ToLower() == ch.ToString() ) {
                hitCount++;
                ChooseHand();
            }
        }    

        gameHitCountOut.text = hitCount.ToString();
    }

    public void Restart() {
        showingResults = false;

        game.SetActive(true);
        results.SetActive(false);

        hitCount = 0;
    }

    public void ShowResults() {
        showingResults = true;

        game.SetActive(false);
        results.SetActive(true);

        resultsHitCountOut.text = hitCount.ToString();
    }
}