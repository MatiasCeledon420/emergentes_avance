using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using TMPro;

public class timer : MonoBehaviour {
    public float target = 0.0f;
    public float init   = 0.0f;
    public TextMeshProUGUI output;

    [SerializeField]
    public UnityEvent OnEnd;

    private bool isRunning = true;

    private float time   = 0.0f;

    void Start() {
        Reset();
    }

    void Update() {
        if ( !isRunning ) { return; }

        time -= Time.deltaTime;

        if ( time <= target ) {
            isRunning = false;

            if ( OnEnd != null ) {
                OnEnd.Invoke();
            }

            // Debug.Log("____END____");
        }

        output.text = time.ToString("0.00");
    }

    public void Reset() {
        time = init;
        isRunning = true;
    }
}