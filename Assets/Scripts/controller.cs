using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

[System.Serializable]
public class HandSign {
    public char letter;
    public GameObject hand;

    public HandSign(HandSign sign) {
        letter = sign.letter;
        hand   = sign.hand;
    }

    public HandSign Clone() {
        return new HandSign(this);
    }
}

public class controller : MonoBehaviour {
    public GameObject  game;
    public TextMeshProUGUI gameHitCountOut;
    public TextMeshProUGUI letterOut;

    public GameObject results;
    public TextMeshProUGUI resultsHitCountOut;

    private int hitCount = 0; 
    private bool showingResults = false;

    [SerializeField] public HandSign[] signs; 
    private HandSign targetSign;

    public void ChooseHand() {
        if ( targetSign != null ) { Destroy(targetSign.hand); }
        targetSign = signs[Random.Range(0, signs.Length)].Clone();
        // , new Vector3(0, 0, 0)
        targetSign.hand = Instantiate(targetSign.hand, game.transform);
        letterOut.text  = targetSign.letter.ToString();
    }

    void Start() {
        ChooseHand();
    }

    void Update() {
        if ( showingResults ) { return; }

        if ( Input.anyKeyDown ) {
            if ( Input.inputString.Length != 1 ) { return; }

            char ch = Input.inputString[0];

            if ( targetSign.letter == ch ) {
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