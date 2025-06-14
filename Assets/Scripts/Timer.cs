using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;
using TMPro;

public class Timer : MonoBehaviour {
    public float target = 0.0f;
    public float init   = 0.0f;
    public GameObject timeBar;

    [SerializeField]
    public UnityEvent OnEnd;

    private bool isRunning = false;
    private float time   = 0.0f;

    void Update() {
        if ( !isRunning ) {
            timeBar.SetActive(false);
            return;
        }

        time -= Time.deltaTime;

        if ( time <= target ) {
            isRunning = false;

            if ( OnEnd != null ) {
                OnEnd.Invoke();
            }
        }

        if ( init != target ) {
            timeBar.transform.Find("Fill").gameObject.GetComponent<Image>().fillAmount = time / ( init - target );
            timeBar.transform.Find("Text").gameObject.GetComponent<TextMeshProUGUI>().text = $"{time.ToString("0.00")}/{target.ToString("0.00")}";
        } else {
            timeBar.SetActive(false);
        }
    }
 
    public void Set(int newInit = int.MaxValue, int newTarget = int.MaxValue) {
        if ( newTarget != int.MaxValue ) { target = newTarget; }
        if ( newInit != int.MaxValue ) { init = newInit; }

        time = init;
        isRunning = true;
    }
}