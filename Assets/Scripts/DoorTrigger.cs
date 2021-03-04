
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class DoorTrigger : UdonSharpBehaviour
{
    Animator animator;

    void Start()
    {
       animator = GetComponentInChildren<Animator>();    
    }

   // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            animator.SetTrigger("isOpen");
        } 
        if (Input.GetKeyDown(KeyCode.T)) 
        {
            animator.ResetTrigger("isOpen");
        }
    }
    private void OnPlayerTriggerEnter(VRCPlayerApi player)
    {
        animator.SetTrigger("isOpen");
     }
    private void OnPlayerTriggerExit(VRCPlayerApi player)
    {
        animator.ResetTrigger("isOpen");
     }
}
